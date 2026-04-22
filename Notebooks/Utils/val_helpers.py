"""
val_helpers.py — Validación técnica de features del modelo nnELM.

Cada función `validar_X` produce:
  (a) figura de verificación técnica por caso testigo (3 paneles estándar)
  (b) figura de distribución global usando el DataFrame completo

Convención de paneles técnicos:
  Col 0: Heatmap de la grilla 24×32 con fijaciones marcadas
  Col 1: Scatter recomputado vs. almacenado en JSON  (debe quedar en y=x)
  Col 2: Evolución temporal del valor a lo largo del scanpath
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

GRID_ROWS, GRID_COLS = 24, 32
CS = 32  # cell_size en píxeles


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers internos
# ═══════════════════════════════════════════════════════════════════════════════

def _unpack(caso):
    """Desempaqueta un caso de CASOS_DATA."""
    return (caso['imagen'], caso['sujeto'], caso['label'],
            caso['mapas'], caso['feats'],
            caso['img'], caso['xs'], caso['ys'], caso['n'])


def _grilla_discreta(mapa_continuo, rows=GRID_ROWS, cols=GRID_COLS, cs=CS):
    """Discretiza mapa continuo (H, W) → grilla (rows, cols) por media de celda."""
    return np.array([
        [mapa_continuo[r*cs:(r+1)*cs, c*cs:(c+1)*cs].mean() for c in range(cols)]
        for r in range(rows)
    ])


def _tabla_verificacion(nombre, fix_rows, fix_cols, vals_recomp, vals_json):
    """Imprime tabla fix | (row,col) | recomputado | json | match."""
    print(f'\n  {"Fix":>3} | {"(row,col)":>9} | {nombre+"_recomp":>16} '
          f'| {nombre+"_json":>16} | match')
    print('  ' + '─' * 65)
    all_ok = True
    for i, (r, c, vr, vj) in enumerate(zip(fix_rows, fix_cols, vals_recomp, vals_json)):
        ok = np.isclose(vr, vj, rtol=1e-3, atol=1e-6)
        all_ok = all_ok and ok
        mark = '✓' if ok else '✗'
        print(f'  {i:>3} | ({int(r):>3},{int(c):>3})   | {vr:>16.6f} '
              f'| {vj:>16.6f} | {mark}')
    status = '✓ Extracción correcta' if all_ok else '✗ Hay discrepancias'
    print(f'  → {status}')
    return all_ok


def _ax_grid_heatmap(ax, grid, fix_rows, fix_cols, vals, titulo='', cmap='inferno'):
    """Grilla 24×32 como heatmap, fijaciones marcadas con índice y valor."""
    im = ax.imshow(grid, cmap=cmap, aspect='auto', origin='upper',
                   interpolation='nearest')
    ax.set_title(titulo, fontsize=9)
    ax.set_xlabel('Columna (grilla 0–31)', fontsize=8)
    ax.set_ylabel('Fila (grilla 0–23)', fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    norm = Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals) + 1e-12)
    cmap_pts = plt.cm.cool

    for i, (r, c, v) in enumerate(zip(fix_rows, fix_cols, vals)):
        r, c = int(r), int(c)
        ax.add_patch(mpatches.Rectangle(
            (c - 0.5, r - 0.5), 1, 1,
            linewidth=1.5, edgecolor='cyan', facecolor='none', zorder=3))
        ax.scatter([c], [r], s=70, c=[cmap_pts(norm(v))],
                   edgecolors='white', lw=0.5, zorder=4)
        ax.text(c, r - 0.38, str(i), ha='center', va='bottom',
                fontsize=6, color='white', fontweight='bold', zorder=5)


def _ax_scatter_verificacion(ax, vals_recomp, vals_json, nombre):
    """Scatter recomputado vs. almacenado; deben caer en la línea y=x."""
    n = len(vals_recomp)
    sc = ax.scatter(vals_recomp, vals_json, s=70, c=range(n), cmap='viridis',
                    edgecolors='k', lw=0.4, zorder=3)
    plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label='orden fijación')

    all_vals = np.concatenate([vals_recomp, vals_json])
    margin = (all_vals.max() - all_vals.min()) * 0.05 + 1e-9
    lim = [all_vals.min() - margin, all_vals.max() + margin]
    ax.plot(lim, lim, 'r--', lw=1.2, label='y = x (ideal)', zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f'Recomputado del grid', fontsize=8)
    ax.set_ylabel(f'Almacenado en JSON', fontsize=8)
    ax.set_title('Verificación: recomputado vs. JSON\n(sobre y=x → extracción correcta)', fontsize=9)

    max_err = np.max(np.abs(vals_recomp - vals_json))
    color = 'green' if max_err < 1e-4 else 'red'
    ax.text(0.05, 0.92, f'Max |Δ| = {max_err:.2e}',
            transform=ax.transAxes, fontsize=8, color=color,
            bbox=dict(facecolor='white', alpha=0.7, pad=2))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _ax_temporal(ax, vals, titulo, color, ylabel, extra_lines=None):
    """Evolución temporal de una feature a lo largo del scanpath."""
    n = len(vals)
    vals_clean = np.where(np.isfinite(vals), vals, np.nan)
    ax.plot(range(n), vals_clean, 'o-', color=color, lw=2, ms=6, zorder=3)
    for i, v in enumerate(vals_clean):
        if np.isfinite(v):
            ax.text(i, v, f' {v:.3f}', va='bottom', ha='left', fontsize=6,
                    color='dimgray')
    if extra_lines:
        for val_h, lbl, col_h in extra_lines:
            ax.axhline(val_h, color=col_h, lw=0.9, ls='--', label=lbl)
        ax.legend(fontsize=7)
    ax.set_xlabel('Fijación', fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(titulo, fontsize=9)
    ax.set_xticks(range(n))
    ax.grid(True, alpha=0.3)


def _global_por_fixacion(df, col, color, titulo_extra=''):
    """Curva media±std por posición de fijación + distribución por target_found."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    grp = df[df['fixation'] < 12].groupby('fixation')[col]
    axes[0].errorbar(grp.mean().index, grp.mean().values, yerr=grp.std().values,
                     fmt='o-', color=color, capsize=3)
    axes[0].set_xlabel('Número de fijación', fontsize=8)
    axes[0].set_ylabel(f'{col} (media ± std)', fontsize=8)
    axes[0].set_title(f'{col} — perfil temporal global', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    for tf, lbl, col_h in [(True, 'Encontrado', 'green'),
                            (False, 'No encontrado', 'tomato')]:
        vals = df[df['target_found'] == tf][col].dropna()
        axes[1].hist(vals, bins=40, alpha=0.55, label=lbl,
                     color=col_h, edgecolor='k', lw=0.3)
    axes[1].set_xlabel(col, fontsize=8)
    axes[1].set_ylabel('Fijaciones', fontsize=8)
    axes[1].set_title(f'Distribución global de {col}', fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    if titulo_extra:
        plt.suptitle(titulo_extra, fontsize=10)
    plt.tight_layout(); plt.show()


def _header(suj, img, lbl, status=''):
    return f'{lbl} — {suj} | {img}' + (f' [{status}]' if status else '')


def _fig_mapas_dinamicos(mapa_stack, fix_rows, fix_cols, vals_escalar,
                          titulo_base, cmap='inferno', suptitle=''):
    """
    Muestra el mapa 2D en las fijaciones de interés: primera, máximo del valor
    escalar, mínimo y última (sin repetir, hasta 4 paneles).

    Args:
        mapa_stack   : (n_fix, 24, 32) — mapa dinámico por fijación
        fix_rows/cols: coordenadas de grilla de cada fijación
        vals_escalar : (n_fix,) — valor escalar asociado (para elegir interesantes)
        titulo_base  : nombre del feature (para el título de cada panel)
        cmap         : colormap
        suptitle     : título global de la figura
    """
    n = mapa_stack.shape[0]
    # Fijaciones de interés: primera, argmax, argmin, última (sin repetir)
    fin_idx = int(np.nanargmin(vals_escalar))
    fmax_idx = int(np.nanargmax(vals_escalar))
    indices = list(dict.fromkeys([0, fmax_idx, fin_idx, n - 1]))  # preserva orden, sin duplicados

    fig, axes = plt.subplots(1, len(indices), figsize=(6 * len(indices), 4.5))
    if len(indices) == 1:
        axes = [axes]

    labels = {0: 'primera', fmax_idx: 'máx', fin_idx: 'mín', n - 1: 'última'}

    for ax, t in zip(axes, indices):
        im = ax.imshow(mapa_stack[t], cmap=cmap, aspect='auto', origin='upper',
                       interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        r, c = int(fix_rows[t]), int(fix_cols[t])
        ax.scatter([c], [r], s=180, c='red', edgecolors='white', lw=1.5, zorder=5,
                   label='fijación actual')
        ax.set_title(
            f't={t} [{labels.get(t, "")}]  fix=({r},{c})\n'
            f'{titulo_base} = {vals_escalar[t]:.4f}',
            fontsize=9)
        ax.set_xlabel('Columna', fontsize=8)
        ax.set_ylabel('Fila', fontsize=8)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    plt.show()


def _ax_scanpath(ax, img, xs, ys, titulo='Imagen + scanpath'):
    """Panel 1: imagen original con scanpath superpuesto."""
    ax.imshow(img)
    n = len(xs)
    for i in range(1, n):
        ax.annotate('', xy=(xs[i], ys[i]), xytext=(xs[i-1], ys[i-1]),
                    arrowprops=dict(arrowstyle='->', color='yellow', lw=1.2, alpha=0.7))
    for i, (x, y) in enumerate(zip(xs, ys)):
        color = 'lime' if i == 0 else 'white'
        ax.add_patch(plt.Circle((x, y), radius=12, color=color, alpha=0.85, zorder=4))
        ax.text(x, y, str(i), ha='center', va='center',
                fontsize=6, color='black', fontweight='bold', zorder=5)
    ax.set_title(titulo, fontsize=9)
    ax.axis('off')


def _ax_mapa_continuo(ax, img, mapa, titulo='Mapa de saliencia', cmap='inferno'):
    """Panel 2: mapa continuo superpuesto sobre la imagen."""
    h, w = img.shape[:2]
    ax.imshow(img, alpha=0.4)
    mapa_norm = (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-12)
    im = ax.imshow(mapa_norm, cmap=cmap, alpha=0.7,
                   extent=[0, w, h, 0], aspect='auto', interpolation='bilinear')
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(titulo, fontsize=9)
    ax.axis('off')


def _ax_fijaciones_por_feature(ax, img, xs, ys, vals, titulo, cmap='plasma'):
    """Paneles 3-5: imagen con fijaciones coloreadas por valor de feature.

    Si len(xs) > len(vals) (e.g. última fijación añadida desde JSON sin feature del
    modelo), las fijaciones extra se dibujan como círculos blancos con borde rojo.
    """
    ax.imshow(img, alpha=0.5)
    n = len(xs)
    norm = Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals) + 1e-12)
    cmap_obj = plt.get_cmap(cmap)
    for i in range(1, n):
        ax.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]],
                '-', color='white', alpha=0.3, lw=1, zorder=2)
    for i, (x, y, v) in enumerate(zip(xs, ys, vals)):
        color = cmap_obj(norm(v))
        ax.add_patch(plt.Circle((x, y), radius=14, color=color, alpha=0.9, zorder=3))
        ax.text(x, y, str(i), ha='center', va='center',
                fontsize=6, color='black', fontweight='bold', zorder=4)
        ax.text(x, y + 18, f'{v:.3f}', ha='center', va='top',
                fontsize=5.5, color='white', zorder=4,
                bbox=dict(facecolor='black', alpha=0.5, pad=1, boxstyle='round'))
    # Fijaciones extra sin feature del modelo (e.g. fijación final en found trials)
    for i in range(len(vals), n):
        x, y = xs[i], ys[i]
        ax.add_patch(plt.Circle((x, y), radius=14, facecolor='none',
                                 edgecolor='red', linewidth=2,
                                 linestyle='--', alpha=0.9, zorder=3))
        ax.text(x, y, str(i), ha='center', va='center',
                fontsize=6, color='red', fontweight='bold', zorder=4)
    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(titulo, fontsize=9)
    ax.axis('off')


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Saliencia
# ═══════════════════════════════════════════════════════════════════════════════

def validar_saliencia(CASOS_DATA, df, ml):
    for caso in CASOS_DATA:
        img_n, suj, lbl, mapas, feats, img_arr, xs, ys, n = _unpack(caso)

        sal_map  = ml.cargar_mapa_saliencia(img_n)
        sal_media   = np.array([f['saliencia_media']   for f in feats['fijaciones']])
        sal_mediana = np.array([f['saliencia_mediana'] for f in feats['fijaciones']])
        sal_pixel   = np.array([f['saliencia_pixel']   for f in feats['fijaciones']])
        n_modelo = len(sal_media)  # fijaciones con datos del modelo (< n si target_found)
        fix_rows = np.array(mapas['fixations_y'][:n_modelo])
        fix_cols = np.array(mapas['fixations_x'][:n_modelo])

        # ── Fig 1: visualización cualitativa (5 paneles) ─────────────────────
        fig, axes = plt.subplots(1, 5, figsize=(28, 5))
        _ax_scanpath(axes[0], img_arr, xs, ys, 'Imagen + scanpath')
        _ax_mapa_continuo(axes[1], img_arr, sal_map, 'Mapa de saliencia\n(DeepGaze II)')
        _ax_fijaciones_por_feature(axes[2], img_arr, xs, ys, sal_media,
                                   'saliencia_media\n(media celda 32px)', cmap='YlOrRd')
        _ax_fijaciones_por_feature(axes[3], img_arr, xs, ys, sal_mediana,
                                   'saliencia_mediana\n(mediana celda 32px)', cmap='YlOrRd')
        _ax_fijaciones_por_feature(axes[4], img_arr, xs, ys, sal_pixel,
                                   'saliencia_pixel\n(valor en píxel exacto)', cmap='YlOrRd')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

        # ── Fig 2: verificación técnica ───────────────────────────────────────
        sal_grid    = _grilla_discreta(sal_map)
        vals_recomp = np.array([sal_grid[int(r), int(c)]
                                 for r, c in zip(fix_rows, fix_cols)])

        print(f'\n── {suj} | {lbl} ──')
        ok = _tabla_verificacion('sal', fix_rows, fix_cols, vals_recomp, sal_media)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        _ax_grid_heatmap(axes[0], sal_grid, fix_rows, fix_cols, vals_recomp,
                         titulo='Grilla saliencia 24×32\n(media de cada celda 32px)')
        _ax_scatter_verificacion(axes[1], vals_recomp, sal_media, 'saliencia_media')
        _ax_temporal(axes[2], sal_media, 'Evolución temporal', 'steelblue', 'saliencia_media',
                     extra_lines=[(sal_grid.mean(), 'media global grilla', 'gray')])
        plt.suptitle(_header(suj, img_n, lbl, '✓' if ok else '✗'), fontsize=10)
        plt.tight_layout(); plt.show()

        # Figura 3: comparacion fijacion a fijacion entre media, mediana y pixel en el mismo grafico
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sal_media, 'o-', label='saliencia_media', color='steelblue')
        ax.plot(sal_mediana, 's-', label='saliencia_mediana', color='darkorange')
        ax.plot(sal_pixel, 'd-', label='saliencia_pixel', color='green')
        ax.set_xlabel('Fijación', fontsize=8)
        ax.set_ylabel('Valor de saliencia', fontsize=8)
        ax.set_title('Comparación de features de saliencia\npor fijación', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    _global_por_fixacion(df, 'saliencia_media', 'steelblue')


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Similitud con el target
# ═══════════════════════════════════════════════════════════════════════════════

def validar_similitud(CASOS_DATA, df, ml):
    for caso in CASOS_DATA:
        img_n, suj, lbl, mapas, feats, img_arr, xs, ys, n = _unpack(caso)

        sim_map     = ml.cargar_mapa_similitud(img_n, mapas['target_stim'])
        sim_media   = np.array([f['similitud_media']   for f in feats['fijaciones']])
        sim_mediana = np.array([f['similitud_mediana'] for f in feats['fijaciones']])
        sim_pixel   = np.array([f['similitud_pixel']   for f in feats['fijaciones']])
        n_modelo = len(sim_media)
        fix_rows = np.array(mapas['fixations_y'][:n_modelo])
        fix_cols = np.array(mapas['fixations_x'][:n_modelo])

        tgt = mapas['target_stim']

        # ── Fig 1: visualización cualitativa (5 paneles) ─────────────────────
        fig, axes = plt.subplots(1, 5, figsize=(28, 5))
        _ax_scanpath(axes[0], img_arr, xs, ys, 'Imagen + scanpath')
        _ax_mapa_continuo(axes[1], img_arr, sim_map,
                          f'Mapa de similitud\n(ResNeXt101 | target: {tgt})')
        _ax_fijaciones_por_feature(axes[2], img_arr, xs, ys, sim_media,
                                   'similitud_media\n(media celda 32px)', cmap='YlGn')
        _ax_fijaciones_por_feature(axes[3], img_arr, xs, ys, sim_mediana,
                                   'similitud_mediana\n(mediana celda 32px)', cmap='YlGn')
        _ax_fijaciones_por_feature(axes[4], img_arr, xs, ys, sim_pixel,
                                   'similitud_pixel\n(valor en píxel exacto)', cmap='YlGn')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

        # ── Fig 2: verificación técnica ───────────────────────────────────────
        sim_grid    = _grilla_discreta(sim_map)
        vals_recomp = np.array([sim_grid[int(r), int(c)]
                                 for r, c in zip(fix_rows, fix_cols)])

        print(f'\n── {suj} | {lbl} | target: {tgt} ──')
        ok = _tabla_verificacion('sim', fix_rows, fix_cols, vals_recomp, sim_media)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        _ax_grid_heatmap(axes[0], sim_grid, fix_rows, fix_cols, vals_recomp,
                         titulo=f'Grilla similitud 24×32\ntarget: {tgt}',
                         cmap='viridis')
        _ax_scatter_verificacion(axes[1], vals_recomp, sim_media, 'similitud_media')
        _ax_temporal(axes[2], sim_media, 'Evolución temporal', 'darkorange', 'similitud_media',
                     extra_lines=[(sim_grid.mean(), 'media global grilla', 'gray')])
        plt.suptitle(_header(suj, img_n, lbl, '✓' if ok else '✗'), fontsize=10)
        plt.tight_layout(); plt.show()

        # ── Fig 3: comparación media / mediana / pixel ────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sim_media,   'o-', label='similitud_media',   color='darkorange')
        ax.plot(sim_mediana, 's-', label='similitud_mediana', color='goldenrod')
        ax.plot(sim_pixel,   'd-', label='similitud_pixel',   color='green')
        ax.set_xlabel('Fijación', fontsize=8)
        ax.set_ylabel('Valor de similitud', fontsize=8)
        ax.set_title('Comparación de features de similitud\npor fijación', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.show()

    _global_por_fixacion(df, 'similitud_media', 'darkorange')


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Evidencia Visual
# ═══════════════════════════════════════════════════════════════════════════════

def validar_evidencia_visual(CASOS_DATA, df):
    for caso in CASOS_DATA:
        img_n, suj, lbl, mapas, feats, img_arr, xs, ys, n = _unpack(caso)

        ve_vals  = np.array([f['evidencia_visual'] for f in feats['fijaciones']])
        n_modelo = len(ve_vals)
        fix_rows = np.array(mapas['fixations_y'][:n_modelo])
        fix_cols = np.array(mapas['fixations_x'][:n_modelo])
        # Recomputar: valor del mapa de evidencia en la celda fijada
        vals_recomp = np.array([mapas['visual_evidence'][i, int(r), int(c)]
                                 for i, (r, c) in enumerate(zip(fix_rows, fix_cols))])

        # ── Fig 1: visualización cualitativa ─────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        _ax_scanpath(axes[0], img_arr, xs, ys, 'Imagen + scanpath')
        _ax_mapa_continuo(axes[1], img_arr, mapas['visual_evidence'][-1],
                          f'visual_evidence — última fijación\n(mapa acumulado 24×32)', cmap='hot')
        _ax_fijaciones_por_feature(axes[2], img_arr, xs, ys, ve_vals,
                                   'evidencia_visual\n(valor en celda fijada)', cmap='hot')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

        _fig_mapas_dinamicos(mapas['visual_evidence'], fix_rows, fix_cols, ve_vals,
                             'evidencia_visual', cmap='hot',
                             suptitle=_header(suj, img_n, lbl) + ' — visual_evidence por fijación')

        # ── Fig 2: verificación técnica ───────────────────────────────────────
        print(f'\n── {suj} | {lbl} ──')
        ok = _tabla_verificacion('ve', fix_rows, fix_cols, vals_recomp, ve_vals)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        _ax_grid_heatmap(axes[0], mapas['visual_evidence'][-1],
                         fix_rows, fix_cols, vals_recomp,
                         titulo=f'visual_evidence en última fijación ({n_modelo-1})\n(fijaciones marcadas con su valor)',
                         cmap='hot')
        _ax_scatter_verificacion(axes[1], vals_recomp, ve_vals, 'evidencia_visual')
        _ax_temporal(axes[2], ve_vals, 'Evolución temporal\n(esperado: monotónica creciente)',
                     'crimson', 'evidencia_visual')
        plt.suptitle(_header(suj, img_n, lbl, '✓' if ok else '✗'), fontsize=10)
        plt.tight_layout(); plt.show()

    _global_por_fixacion(df, 'evidencia_visual', 'crimson')


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Competencia Foveal-Periférica
# ═══════════════════════════════════════════════════════════════════════════════

def validar_competencia_fp(CASOS_DATA, df):
    for caso in CASOS_DATA:
        img_n, suj, lbl, mapas, feats, img_arr, xs, ys, n = _unpack(caso)

        fp_var = np.array([f['competencia_fp_var']   for f in feats['fijaciones']])
        fp_max = np.array([f['competencia_fp_max']   for f in feats['fijaciones']])
        fp_med = np.array([f['competencia_fp_media'] for f in feats['fijaciones']])
        n_fp   = len(fp_var)
        fix_rows = np.array(mapas['fixations_y'][:n_fp])
        fix_cols = np.array(mapas['fixations_x'][:n_fp])

        # ── Fig 1: visualización cualitativa ─────────────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        _ax_scanpath(axes[0], img_arr, xs, ys, 'Imagen + scanpath')
        _ax_fijaciones_por_feature(axes[1], img_arr, xs, ys, fp_var,
                                   'competencia_fp_var', cmap='Purples')
        _ax_fijaciones_por_feature(axes[2], img_arr, xs, ys, fp_max,
                                   'competencia_fp_max', cmap='Oranges')
        _ax_fijaciones_por_feature(axes[3], img_arr, xs, ys, fp_med,
                                   'competencia_fp_media', cmap='Blues')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

        # Posterior subyacente — fp mide concentración en región foveal/periférica
        _fig_mapas_dinamicos(mapas['posterior'][:n_fp], fix_rows, fix_cols, fp_var,
                             'competencia_fp_var', cmap='viridis',
                             suptitle=_header(suj, img_n, lbl) + ' — posterior (base de fp) por fijación')

        # ── Fig 2: evolución temporal ─────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        for ax, vals, lbl_f, col in [
            (axes[0], fp_var, 'competencia_fp_var',   'purple'),
            (axes[1], fp_max, 'competencia_fp_max',   'darkorange'),
            (axes[2], fp_med, 'competencia_fp_media', 'steelblue'),
        ]:
            _ax_temporal(ax, vals, f'{lbl_f}\npor fijación', col, lbl_f)
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

    # Global: distribución por target_found para fp_var y fp_max
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col_name in [(axes[0], 'competencia_fp_var'), (axes[1], 'competencia_fp_max')]:
        for tf, lbl_g, col in [(True, 'Encontrado', 'green'), (False, 'No encontrado', 'tomato')]:
            v = df[df['target_found'] == tf][col_name].dropna()
            ax.hist(v, bins=40, alpha=0.55, label=lbl_g, color=col, edgecolor='k', lw=0.3)
        ax.set_xlabel(col_name, fontsize=8); ax.set_ylabel('Fijaciones', fontsize=8)
        ax.set_title(f'Distribución global de {col_name}', fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Posterior
# ═══════════════════════════════════════════════════════════════════════════════

def validar_posterior(CASOS_DATA, df, ml):
    for caso in CASOS_DATA:
        img_n, suj, lbl, mapas, feats, img_arr, xs, ys, n = _unpack(caso)

        post_vals = np.array([f['posterior'] for f in feats['fijaciones']])
        n_modelo  = len(post_vals)
        fix_rows  = np.array(mapas['fixations_y'][:n_modelo])
        fix_cols  = np.array(mapas['fixations_x'][:n_modelo])
        vals_recomp = np.array([mapas['posterior'][i, int(r), int(c)]
                                  for i, (r, c) in enumerate(zip(fix_rows, fix_cols))])

        post_max = ml.concentracion_posterior(mapas)

        # ── Fig 1: visualización cualitativa ─────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        _ax_scanpath(axes[0], img_arr, xs, ys, 'Imagen + scanpath')
        _ax_mapa_continuo(axes[1], img_arr, mapas['posterior'][-1],
                          f'Posterior — última fijación\n(mapa 24×32)', cmap='viridis')
        _ax_fijaciones_por_feature(axes[2], img_arr, xs, ys, post_vals,
                                   'posterior\n(valor en celda fijada)', cmap='viridis')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

        _fig_mapas_dinamicos(mapas['posterior'], fix_rows, fix_cols, post_vals,
                             'posterior', cmap='viridis',
                             suptitle=_header(suj, img_n, lbl) + ' — posterior por fijación')

        # ── Fig 2: verificación técnica ───────────────────────────────────────
        print(f'\n── {suj} | {lbl} ──')
        ok = _tabla_verificacion('post', fix_rows, fix_cols, vals_recomp, post_vals)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        _ax_grid_heatmap(axes[0], mapas['posterior'][-1],
                         fix_rows, fix_cols, vals_recomp,
                         titulo=f'Posterior en última fijación ({n_modelo-1})',
                         cmap='viridis')
        _ax_scatter_verificacion(axes[1], vals_recomp, post_vals, 'posterior')
        _ax_temporal(axes[2], post_vals, 'Posterior en celda fijada vs. máximo',
                     'navy', 'posterior',
                     extra_lines=[(pm, f'max_mapa t={i}', 'gray')
                                  for i, pm in enumerate(post_max)])
        plt.suptitle(_header(suj, img_n, lbl, '✓' if ok else '✗'), fontsize=10)
        plt.tight_layout(); plt.show()

    _global_por_fixacion(df, 'posterior', 'navy')


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Entropía del Prior
# ═══════════════════════════════════════════════════════════════════════════════

def validar_prior(CASOS_DATA, df):
    for caso in CASOS_DATA:
        _, suj, lbl, _, feats, _, _, _, _ = _unpack(caso)
        prior_sal = [f['entropia_prior_saliencia'] for f in feats['fijaciones']]
        prior_mod = [f['entropia_prior_modelo']    for f in feats['fijaciones']]
        # Debe ser constante dentro del trial (una sola imagen → un solo prior)
        print(f'{suj} | entropia_prior_saliencia: {set(round(x,6) for x in prior_sal)}')
        print(f'{suj} | entropia_prior_modelo:    {set(round(x,6) for x in prior_mod)}')

    prior_by_img = (df.groupby('imagen')[['entropia_prior_saliencia', 'entropia_prior_modelo']]
                      .first().reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, col_h in [
        (axes[0], 'entropia_prior_saliencia', 'steelblue'),
        (axes[1], 'entropia_prior_modelo',    'darkorange'),
    ]:
        axes_col = ax
        axes_col.hist(prior_by_img[col], bins=30, color=col_h, edgecolor='k', lw=0.3)
        axes_col.axvline(prior_by_img[col].mean(), color='red', ls='--',
                         label=f'media={prior_by_img[col].mean():.3f}')
        axes_col.set_xlabel(col, fontsize=8); axes_col.set_ylabel('Imágenes', fontsize=8)
        axes_col.set_title(f'Distribución de {col}\n(1 valor por imagen)', fontsize=9)
        axes_col.legend(fontsize=8); axes_col.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Entropía del Posterior
# ═══════════════════════════════════════════════════════════════════════════════

def validar_entropia(CASOS_DATA, df):
    for caso in CASOS_DATA:
        img_n, suj, lbl, mapas, feats, img_arr, xs, ys, n = _unpack(caso)

        ent_vals = np.array([f['entropia_posterior'] for f in feats['fijaciones']])
        n_modelo = len(ent_vals)
        fix_rows = np.array(mapas['fixations_y'][:n_modelo])
        fix_cols = np.array(mapas['fixations_x'][:n_modelo])
        # Recomputar: H = -sum(p * log(p)) sumado sobre celdas
        eps = 1e-12
        vals_recomp = np.array([
            -np.sum(mapas['posterior'][i] * np.log(mapas['posterior'][i] + eps))
            for i in range(n_modelo)
        ])

        # ── Fig 1: visualización cualitativa ─────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        _ax_scanpath(axes[0], img_arr, xs, ys, 'Imagen + scanpath')
        _ax_mapa_continuo(axes[1], img_arr, mapas['entropy_map'][-1],
                          'entropy_map — última fijación\n(-p·log(p) por celda)', cmap='magma')
        _ax_fijaciones_por_feature(axes[2], img_arr, xs, ys, ent_vals,
                                   'entropia_posterior\n(entropía escalar del posterior)', cmap='magma')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

        _fig_mapas_dinamicos(mapas['entropy_map'], fix_rows, fix_cols, ent_vals,
                             'entropia_posterior', cmap='magma',
                             suptitle=_header(suj, img_n, lbl) + ' — entropy_map por fijación')

        # ── Fig 2: verificación técnica ───────────────────────────────────────
        print(f'\n── {suj} | {lbl} ──')
        max_err = np.max(np.abs(vals_recomp - ent_vals))
        print(f'  Max |Δ| entropia recomputada vs JSON: {max_err:.2e} '
              f'{"✓" if max_err < 1e-3 else "✗"}')

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        _ax_scatter_verificacion(axes[0], vals_recomp, ent_vals, 'entropia_posterior')
        _ax_temporal(axes[1], ent_vals,
                     f'Evolución temporal\n(esperado: decreciente, found={feats["target_found"]})',
                     'tomato', 'entropia_posterior')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

    _global_por_fixacion(df, 'entropia_posterior', 'tomato')


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Ganancia de Información (KL Divergencia)
# ═══════════════════════════════════════════════════════════════════════════════

def validar_kl(CASOS_DATA, df):
    for caso in CASOS_DATA:
        img_n, suj, lbl, mapas, feats, img_arr, xs, ys, n = _unpack(caso)

        kl_vals = np.array([f['ganancia_informacion'] if f['ganancia_informacion'] is not None
                             else np.nan for f in feats['fijaciones']])
        n_modelo = len(kl_vals)
        # Recomputar: KL(P_t || P_{t-1}) — fijación 0 = NaN
        eps = 1e-12
        post = mapas['posterior']
        kl_recomp = np.full(n_modelo, np.nan)
        for i in range(1, n_modelo):
            kl_recomp[i] = np.sum(
                post[i] * np.log((post[i] + eps) / (post[i-1] + eps))
            )

        # ── Fig 1: visualización cualitativa ─────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        _ax_scanpath(axes[0], img_arr, xs, ys, 'Imagen + scanpath')
        _ax_fijaciones_por_feature(axes[1], img_arr, xs, ys, kl_vals,
                                   'ganancia_informacion (KL)\n(fijación 0 = NaN)', cmap='YlOrBr')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

        # ── Fig 2: verificación técnica ───────────────────────────────────────
        valid = ~np.isnan(kl_vals)
        max_err = np.max(np.abs(kl_recomp[valid] - kl_vals[valid]))
        print(f'\n── {suj} | {lbl} | Max |Δ| KL: {max_err:.2e} '
              f'{"✓" if max_err < 1e-4 else "✗"}')

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        _ax_scatter_verificacion(axes[0], kl_recomp[valid], kl_vals[valid], 'ganancia_informacion')
        _ax_temporal(axes[1], kl_vals, 'KL por fijación\n(fijación 0 = NaN por definición)',
                     'goldenrod', 'ganancia_informacion (KL)')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

    # Global
    kl_all = df[df['fixation'] > 0]['ganancia_informacion'].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(kl_all.clip(upper=kl_all.quantile(0.99)), bins=50,
                 color='goldenrod', edgecolor='k', lw=0.3)
    axes[0].axvline(0, color='red', ls='--', lw=1.5, label='KL=0')
    axes[0].set_xlabel('ganancia_informacion', fontsize=8)
    axes[0].set_ylabel('Fijaciones', fontsize=8)
    axes[0].set_title(f'Distribución global (fij > 0)\nmedia={kl_all.mean():.4f}  '
                      f'neg={(kl_all < 0).sum()} (esperado: 0)', fontsize=9)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    grp = df[df['fixation'].between(1, 11)].groupby('fixation')['ganancia_informacion']
    axes[1].errorbar(grp.mean().index, grp.mean().values, yerr=grp.std().values,
                     fmt='o-', color='goldenrod', capsize=3)
    axes[1].set_xlabel('Número de fijación', fontsize=8)
    axes[1].set_ylabel('KL (media ± std)', fontsize=8)
    axes[1].set_title('KL promedio por posición de fijación', fontsize=9)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Gap de Entropía con el Óptimo (EIG)
# ═══════════════════════════════════════════════════════════════════════════════

def validar_eig(CASOS_DATA, df, CELL_SIZE=32):
    for caso in CASOS_DATA:
        img_n, suj, lbl, mapas, feats, img_arr, xs, ys, n = _unpack(caso)

        eig_fix  = np.array([f['eig_en_fijacion'] if f['eig_en_fijacion'] is not None
                              else np.nan for f in feats['fijaciones']])
        eig_opt  = np.array([f['eig_optimo'] for f in feats['fijaciones']])
        gap_vals = np.array([f['gap_entropia'] if f['gap_entropia'] is not None
                              else np.nan for f in feats['fijaciones']])

        # Recomputar eig_en_fijacion[t] = EIG_map[t, r_{t+1}, c_{t+1}]
        # n_modelo = fijaciones con datos del modelo; n puede ser n_modelo+1 (found trial)
        n_modelo = len(eig_fix)
        fix_rows = np.array(mapas['fixations_y'][:n_modelo])
        fix_cols = np.array(mapas['fixations_x'][:n_modelo])
        eig_recomp = np.full(n_modelo, np.nan)
        eig_opt_recomp = np.full(n_modelo, np.nan)
        for t in range(n_modelo - 1):
            eig_map_t = mapas['expected_ig_map'][t]
            eig_recomp[t]     = eig_map_t[int(fix_rows[t+1]), int(fix_cols[t+1])]
            eig_opt_recomp[t] = eig_map_t.max()

        valid = ~np.isnan(eig_fix)
        err_fix = np.max(np.abs(eig_recomp[valid] - eig_fix[valid]))
        err_opt = np.max(np.abs(eig_opt_recomp[valid] - eig_opt[valid]))
        print(f'\n── {suj} | {lbl} ──')
        print(f'  eig_en_fijacion  Max|Δ|: {err_fix:.2e} {"✓" if err_fix < 1e-4 else "✗"}')
        print(f'  eig_optimo       Max|Δ|: {err_opt:.2e} {"✓" if err_opt < 1e-4 else "✗"}')

        # ── Fig 1: visualización cualitativa ─────────────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        _ax_scanpath(axes[0], img_arr, xs, ys, 'Imagen + scanpath')
        _ax_mapa_continuo(axes[1], img_arr, mapas['expected_ig_map'][-1],
                          'EIG map — última fijación\n(ganancia esperada por celda)', cmap='cividis')
        _ax_fijaciones_por_feature(axes[2], img_arr, xs, ys, eig_fix,
                                   'eig_en_fijacion\n(EIG donde fijó el humano t+1)', cmap='cividis')
        _ax_fijaciones_por_feature(axes[3], img_arr, xs, ys, gap_vals,
                                   'gap_entropia\n(EIG_óptimo − EIG_humano)', cmap='RdYlGn_r')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

        _fig_mapas_dinamicos(mapas['expected_ig_map'], fix_rows, fix_cols, eig_fix,
                             'eig_en_fijacion', cmap='cividis',
                             suptitle=_header(suj, img_n, lbl) + ' — EIG map por fijación')

        # ── Fig 2: verificación técnica ───────────────────────────────────────
        step = min(2, n_modelo - 2)
        eig_map_step = mapas['expected_ig_map'][step]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 0: EIG map como grilla con fijación actual y próxima marcadas
        im = axes[0].imshow(eig_map_step, cmap='cividis', aspect='auto', origin='upper',
                            interpolation='nearest')
        plt.colorbar(im, ax=axes[0], fraction=0.04)
        opt_r, opt_c = np.unravel_index(eig_map_step.argmax(), eig_map_step.shape)
        axes[0].scatter([fix_cols[step]],   [fix_rows[step]],   s=150, c='white',
                        edgecolors='gray', lw=1.5, zorder=4, label=f'actual t={step}')
        axes[0].scatter([fix_cols[step+1]], [fix_rows[step+1]], s=150, c='red',
                        edgecolors='white', lw=1.5, zorder=5, label=f'humano t+1={step+1}')
        axes[0].scatter([opt_c], [opt_r], s=200, c='lime', marker='*',
                        edgecolors='white', lw=1, zorder=5, label='argmax EIG (óptimo)')
        axes[0].legend(fontsize=7, loc='lower right')
        axes[0].set_title(f'EIG map en t={step}\n(col=argmax, rojo=humano t+1)', fontsize=9)
        axes[0].set_xlabel('Columna'); axes[0].set_ylabel('Fila')

        # Panel 1: scatter eig_en_fijacion recomp vs json
        _ax_scatter_verificacion(axes[1], eig_recomp[valid], eig_fix[valid], 'eig_en_fijacion')

        # Panel 2: eig_optimo vs eig_en_fijacion temporal
        _ax_temporal(axes[2], eig_opt, 'EIG óptimo vs. donde fue el humano', 'gray', 'EIG')
        axes[2].plot(np.where(valid)[0], eig_fix[valid], 'o-', color='mediumseagreen',
                     lw=2, label='EIG en fijación t+1 (humano)')
        axes[2].fill_between(np.where(valid)[0], eig_fix[valid], eig_opt[valid],
                             alpha=0.2, color='red', label='gap')
        axes[2].legend(fontsize=7)

        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

    _global_por_fixacion(df, 'gap_entropia', 'salmon')


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Distancia con la Decisión del Modelo
# ═══════════════════════════════════════════════════════════════════════════════

def validar_distancia(CASOS_DATA, df, CELL_SIZE=32):
    for caso in CASOS_DATA:
        img_n, suj, lbl, mapas, feats, img_arr, xs, ys, n = _unpack(caso)

        dist_vals = np.array([f['distancia_grilla'] if f['distancia_grilla'] is not None
                               else np.nan for f in feats['fijaciones']])
        mod_ys_g  = np.array([f['modelo_fix_y'] for f in feats['fijaciones']])
        mod_xs_g  = np.array([f['modelo_fix_x'] for f in feats['fijaciones']])

        # Recomputar: dist_recomp[t] = ||argmax(EIG_t) - (r_{t+1}, c_{t+1})||
        n_modelo = len(dist_vals)
        fix_rows = np.array(mapas['fixations_y'][:n_modelo])
        fix_cols = np.array(mapas['fixations_x'][:n_modelo])
        dist_recomp = np.full(n_modelo, np.nan)
        mod_r_recomp = np.full(n_modelo, np.nan)
        mod_c_recomp = np.full(n_modelo, np.nan)
        for t in range(n_modelo - 1):
            eig_map_t = mapas['expected_ig_map'][t]
            opt_r, opt_c = np.unravel_index(eig_map_t.argmax(), eig_map_t.shape)
            mod_r_recomp[t], mod_c_recomp[t] = opt_r, opt_c
            dist_recomp[t] = np.sqrt((fix_rows[t+1] - opt_r)**2 +
                                      (fix_cols[t+1] - opt_c)**2)

        valid = ~np.isnan(dist_vals)
        err = np.max(np.abs(dist_recomp[valid] - dist_vals[valid]))
        err_y = np.max(np.abs(mod_r_recomp[valid] - mod_ys_g[valid]))
        err_x = np.max(np.abs(mod_c_recomp[valid] - mod_xs_g[valid]))
        print(f'\n── {suj} | {lbl} ──')
        print(f'  distancia_grilla  Max|Δ|: {err:.2e}    {"✓" if err < 1e-3 else "✗"}')
        print(f'  modelo_fix_y      Max|Δ|: {err_y:.2e}  {"✓" if err_y < 1e-3 else "✗"}')
        print(f'  modelo_fix_x      Max|Δ|: {err_x:.2e}  {"✓" if err_x < 1e-3 else "✗"}')

        # ── Fig 1: visualización cualitativa ─────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        _ax_scanpath(axes[0], img_arr, xs, ys, 'Imagen + scanpath')
        _ax_fijaciones_por_feature(axes[1], img_arr, xs, ys, dist_vals,
                                   'distancia_grilla\n(celdas entre humano y argmax EIG)', cmap='RdYlGn_r')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

        # EIG map subyacente — distancia = ||argmax(EIG_t) − fix_{t+1}||
        valid_dist = ~np.isnan(dist_vals)
        if valid_dist.any():
            _fig_mapas_dinamicos(mapas['expected_ig_map'][:n_modelo], fix_rows, fix_cols,
                                 np.where(valid_dist, dist_vals, 0.0),
                                 'distancia_grilla', cmap='cividis',
                                 suptitle=_header(suj, img_n, lbl) + ' — EIG map (base de distancia) por fijación')

        # ── Fig 2: verificación técnica ───────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Panel 0: scanpaths humano vs modelo como grilla
        ax = axes[0]
        ax.set_xlim(-0.5, GRID_COLS - 0.5); ax.set_ylim(GRID_ROWS - 0.5, -0.5)
        ax.set_facecolor('#1a1a2e')
        ax.set_title('Scanpath humano vs. recomendaciones modelo\n(espacio grilla 24×32)', fontsize=9)
        ax.set_xlabel('Columna'); ax.set_ylabel('Fila')
        for t in range(n_modelo - 1):
            ax.plot([fix_cols[t], fix_cols[t+1]], [fix_rows[t], fix_rows[t+1]],
                    '-', color='tomato', lw=1.2, alpha=0.7)
        ax.scatter(fix_cols, fix_rows, s=80, c='white', zorder=5,
                   edgecolors='tomato', lw=1.2)
        for t in range(n_modelo):
            ax.text(fix_cols[t], fix_rows[t] - 0.4, str(t),
                    ha='center', fontsize=6, color='white', fontweight='bold')
        for t in range(n_modelo - 1):
            ax.plot([mod_xs_g[t], mod_xs_g[t+1]], [mod_ys_g[t], mod_ys_g[t+1]],
                    '-', color='lime', lw=1.2, alpha=0.7)
        ax.scatter(mod_xs_g[valid], mod_ys_g[valid], s=80, c='black', zorder=5,
                   edgecolors='lime', lw=1.2, marker='D')
        for t in range(n_modelo - 1):
            ax.annotate('', xy=(mod_xs_g[t], mod_ys_g[t]),
                        xytext=(fix_cols[t+1], fix_rows[t+1]),
                        arrowprops=dict(arrowstyle='->', color='yellow', lw=0.8, alpha=0.5))
        ax.legend(handles=[mpatches.Patch(color='tomato', label='Humano'),
                           mpatches.Patch(color='lime',   label='Modelo (argmax EIG)')],
                  fontsize=8, loc='lower right')

        # Panel 1: distancia por fijación
        _ax_temporal(axes[1], dist_vals, 'Distancia humano vs. modelo por fijación\n(celdas de grilla)',
                     'darkorchid', 'distancia_grilla (celdas)')

        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

    # Global
    dist_all = df['distancia_grilla'].dropna()
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(dist_all, bins=40, color='darkorchid', edgecolor='k', lw=0.3)
    ax.axvline(dist_all.mean(), color='red', lw=1.5, ls='--',
               label=f'media={dist_all.mean():.2f} celdas = {dist_all.mean()*CELL_SIZE:.0f}px')
    ax.set_xlabel('distancia_grilla (celdas)', fontsize=8)
    ax.set_ylabel('Fijaciones', fontsize=8)
    ax.set_title('Distribución global de distancia_grilla', fontsize=9)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    for tf, lbl_g in [(True, 'Encontrado'), (False, 'No encontrado')]:
        m = df[df['target_found'] == tf]['distancia_grilla'].mean()
        print(f'Distancia media ({lbl_g}): {m:.2f} celdas = {m*CELL_SIZE:.0f}px')


# ═══════════════════════════════════════════════════════════════════════════════
# 11-12. Competencia en el Mapa de EIG
# ═══════════════════════════════════════════════════════════════════════════════

def validar_competencia_eig(CASOS_DATA, df):
    for caso in CASOS_DATA:
        img_n, suj, lbl, mapas, feats, img_arr, xs, ys, n = _unpack(caso)

        ratio_vals = np.array([f['competencia_ratio']         for f in feats['fijaciones']])
        dens_vals  = np.array([f['competencia_densidad_norm'] for f in feats['fijaciones']])
        dens_abs   = np.array([f['competencia_densidad']      for f in feats['fijaciones']])
        n_modelo   = len(ratio_vals)
        fix_rows   = np.array(mapas['fixations_y'][:n_modelo])
        fix_cols   = np.array(mapas['fixations_x'][:n_modelo])

        # Recomputar según definición en 3-creacion-features.ipynb
        # ratio         = segundo_mayor / mayor       (qué tan "cerca" está el 2° candidato)
        # densidad      = count(celdas >= 95% del max)
        # densidad_norm = densidad / total_celdas
        ratio_recomp = np.zeros(n_modelo)
        dens_recomp  = np.zeros(n_modelo)
        dens_abs_recomp = np.zeros(n_modelo)
        for i in range(n_modelo):
            eig_map = mapas['expected_ig_map'][i]
            flat = eig_map.flatten()
            sorted_vals = np.sort(flat)[::-1]
            mayor = sorted_vals[0]
            ratio_recomp[i] = float(sorted_vals[1] / mayor) if mayor > 1e-12 else 0.0
            umbral = mayor * 0.95
            cnt = int(np.sum(eig_map >= umbral))
            dens_abs_recomp[i] = cnt
            dens_recomp[i]     = cnt / eig_map.size

        err_r  = np.max(np.abs(ratio_recomp    - ratio_vals))
        err_d  = np.max(np.abs(dens_recomp     - dens_vals))
        err_da = np.max(np.abs(dens_abs_recomp - dens_abs))
        print(f'\n── {suj} | {lbl} ──')
        print(f'  competencia_ratio         Max|Δ|: {err_r:.2e}  {"✓" if err_r  < 1e-4 else "✗"}')
        print(f'  competencia_densidad_norm Max|Δ|: {err_d:.2e}  {"✓" if err_d  < 1e-4 else "✗"}')
        print(f'  competencia_densidad      Max|Δ|: {err_da:.2e} {"✓" if err_da < 1e-4 else "✗"}')

        # ── Fig 1: visualización cualitativa ─────────────────────────────────
        fig, axes = plt.subplots(1, 4, figsize=(24, 5))
        _ax_scanpath(axes[0], img_arr, xs, ys, 'Imagen + scanpath')
        _ax_fijaciones_por_feature(axes[1], img_arr, xs, ys, ratio_vals,
                                   'competencia_ratio\n(EIG_2° / EIG_1°)', cmap='RdYlGn_r')
        _ax_fijaciones_por_feature(axes[2], img_arr, xs, ys, dens_abs,
                                   'competencia_densidad\n(# celdas ≥95% max)', cmap='RdYlGn_r')
        _ax_fijaciones_por_feature(axes[3], img_arr, xs, ys, dens_vals,
                                   'competencia_densidad_norm\n(celdas ≥95% max / total)', cmap='RdYlGn_r')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

        _fig_mapas_dinamicos(mapas['expected_ig_map'][:n_modelo], fix_rows, fix_cols, ratio_vals,
                             'competencia_ratio', cmap='cividis',
                             suptitle=_header(suj, img_n, lbl) + ' — EIG map (base de competencia) por fijación')

        # ── Fig 2: evolución temporal de ratio y densidad ────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        _ax_temporal(axes[0], ratio_vals,
                     'competencia_ratio por fijación\n(2°/1° EIG — alto = ambigüedad)',
                     'steelblue', 'ratio (2°/1°)')
        _ax_temporal(axes[1], dens_abs,
                     'competencia_densidad\n(# celdas ≥95% del máximo EIG)',
                     'darkorange', 'densidad (celdas)')
        axes[1].plot(range(n_modelo), dens_abs * (768 / dens_abs.max() + 1e-12),
                     '--', color='gray', lw=1, alpha=0.5)  # referencia visual
        _ax_temporal(axes[2], dens_vals,
                     'densidad vs. densidad_norm\n(absoluta vs. proporción)',
                     'darkorchid', 'densidad_norm')
        # Superponer densidad_norm en el mismo eje
        ax2b = axes[2].twinx()
        ax2b.plot(range(n_modelo), dens_abs, 's--', color='darkorange',
                  lw=1.5, ms=5, alpha=0.7, label='densidad (abs)')
        ax2b.set_ylabel('densidad (# celdas)', fontsize=8, color='darkorange')
        ax2b.tick_params(axis='y', labelcolor='darkorange')
        ax2b.legend(fontsize=7, loc='upper right')
        plt.suptitle(_header(suj, img_n, lbl), fontsize=10)
        plt.tight_layout(); plt.show()

        # ── Fig 3: verificación técnica ───────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        _ax_scatter_verificacion(axes[0], ratio_recomp,    ratio_vals, 'competencia_ratio')
        _ax_scatter_verificacion(axes[1], dens_recomp,     dens_vals,  'competencia_densidad_norm')
        _ax_scatter_verificacion(axes[2], dens_abs_recomp, dens_abs,   'competencia_densidad')
        plt.suptitle(_header(suj, img_n, lbl, 'verificación'), fontsize=10)
        plt.tight_layout(); plt.show()

        # EIG map de la primera fijación con top-2 marcados
        eig_map_0 = mapas['expected_ig_map'][0]
        flat = eig_map_0.flatten()
        top2_idx = np.argpartition(flat, -2)[-2:]
        top2 = sorted([np.unravel_index(i, eig_map_0.shape) for i in top2_idx],
                      key=lambda c: eig_map_0[c], reverse=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(eig_map_0, cmap='cividis', aspect='auto', origin='upper',
                       interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.04)
        for rank, (r, c) in enumerate(top2):
            ax.scatter([c], [r], s=200, c=['red', 'lime'][rank], marker=['*', 'D'][rank],
                       edgecolors='white', lw=1.2, zorder=5,
                       label=f'{"1°" if rank==0 else "2°"} mayor EIG')
        ax.legend(fontsize=8)
        ax.set_title(f'EIG map t=0\nratio=2°/1°={ratio_vals[0]:.3f} | densidad={int(dens_abs[0])} celdas', fontsize=9)
        ax.set_xlabel('Columna'); ax.set_ylabel('Fila')
        plt.tight_layout(); plt.show()

    # Global
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, col, col_color in [
        (axes[0], 'competencia_ratio',         'steelblue'),
        (axes[1], 'competencia_densidad',      'darkorange'),
        (axes[2], 'competencia_densidad_norm', 'darkorchid'),
    ]:
        grp = df[df['fixation'] < 12].groupby('fixation')[col]
        ax.errorbar(grp.mean().index, grp.mean().values, yerr=grp.std().values,
                    fmt='o-', color=col_color, capsize=3)
        ax.set_xlabel('Número de fijación', fontsize=8)
        ax.set_ylabel(f'{col} (media ± std)', fontsize=8)
        ax.set_title(f'{col} — perfil temporal global', fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()
