# External imports
import numpy as np
from scipy.stats import logistic, norm, skewnorm
import pygame
import time
import matplotlib.pyplot as plt
import sys
import pandas as pd
import seaborn as sns
import os

#Internal imports
from Model_ALL_RandomWalk import generate_random_walk, fit_random_walk, get_n_steps, get_proposal_distribution

############################################################################################################################################
"""Cache"""

RW_CACHE = {
    "LT": fit_random_walk("LT"),
    "CAM": fit_random_walk("CAM"),
    "LLS_B": fit_random_walk("LLS_B"),
    "LLS_A": fit_random_walk("LLS_A")}

RW_PROPOSAL_STD = {
    "LT": get_proposal_distribution("LT"),
    "CAM": get_proposal_distribution("CAM"),
    "LLS_B": get_proposal_distribution("LLS_B"),
    "LLS_A": get_proposal_distribution("LLS_A")}

STEP_CACHE = {
    "LT": get_n_steps("LT"),
    "CAM": get_n_steps("CAM"),
    "LLS_B": get_n_steps("LLS_B"),
    "LLS_A": get_n_steps("LLS_A")}

params = {
    "LT": (-0.93, 0.06),
    "CAM": (0.29, 0.22),
    "LLSB": (-0.08, 0.06),
    "LLSA": (-0.25, 0.07)}

def generate_rw(sensor, n_steps, target_dist, params):
    proposal_std = RW_PROPOSAL_STD[sensor]

    return generate_random_walk(
        sensor,
        n_steps,
        proposal_std,     # ✅ real step size (important)
        target_dist,      # ✅ YOUR distribution
        norm,             # ✅ acts as base distribution
        params            # ✅ used for start value
    )

normal_mode = False

def NORMAL_distribution(mu, sigma):
    return lambda x: norm.pdf(x, loc=mu, scale=sigma)

def LT_distribution(mean_shift, scale_factor):
    mu = -0.93 + mean_shift
    s = 0.036 * scale_factor
    return lambda x: logistic.pdf(x, loc=mu, scale=s)

def CAM_distribution(mean_shift, scale_factor):
    xi = 0.53 + mean_shift
    omega = 0.32 * scale_factor
    alpha = -2.29
    return lambda x: skewnorm.pdf(x, a=alpha, loc=xi, scale=omega)

def LLSB_distribution(mean_shift, scale_factor):
    mu = mean_shift
    sigma = scale_factor
    return lambda x: norm.pdf(x, loc=mu, scale=sigma)

def LLSA_distribution(mean_shift, scale_factor):
    mu = -0.25 + mean_shift
    s = 0.041 * scale_factor
    return lambda x: logistic.pdf(x, loc=mu, scale=s)

LT_steps, LT_std, LT_target, LT_dist, LT_params = RW_CACHE["LT"]
CAM_steps, CAM_std, CAM_target, CAM_dist, CAM_params = RW_CACHE["CAM"]
LLSB_steps, LLSB_std, LLSB_target, LLSB_dist, LLSB_params = RW_CACHE["LLS_B"]
LLSA_steps, LLSA_std, LLSA_target, LLSA_dist, LLSA_params = RW_CACHE["LLS_A"]

LT = generate_rw("LT",LT_steps, NORMAL_distribution(*params["LT"]) if normal_mode else LT_distribution(*params["LT"]), params["LT"])
CAM = generate_rw("CAM",CAM_steps, NORMAL_distribution(*params["CAM"]) if normal_mode else CAM_distribution(*params["CAM"]), params["CAM"])
LLSB = generate_rw("LLS_B", LLSB_steps, NORMAL_distribution(*params["LLSB"]) if normal_mode else LLSB_distribution(*params["LLSB"]), params["LLSB"])
LLSA = generate_rw("LLS_A", LLSA_steps, NORMAL_distribution(*params["LLSA"]) if normal_mode else LLSA_distribution(*params["LLSA"]), params["LLSA"])

############################################################################################################################################
"""Help Functions"""

def LT_parameters(mean_shift, scale_factor):
    mu = -0.93 + mean_shift
    s = 0.036 * scale_factor
    return mu, s

def LLSA_parameters(mean_shift, scale_factor):
    mu = -0.25 + mean_shift
    s = 0.041 * scale_factor
    return mu, s

def LLSB_parameters(mean_shift, scale_factor):
    mu = mean_shift
    sigma = scale_factor
    return mu, sigma

def CAM_parameters(mean_shift, scale_factor):
    xi = 0.53 + mean_shift
    omega = 0.32 * scale_factor
    alpha = -2.29
    return xi, omega

def draw_single_distribution(screen, dist_func, rect, color, title, params=None):
    global normal_mode
    x0, y0, w, h = rect

    pygame.draw.rect(screen, (80, 80, 80), rect, 2)

    # -----------------------------
    # FIXED DOMAIN
    # -----------------------------
    x_min, x_max = -1.2, 1.2
    x_vals = np.linspace(x_min, x_max, 400)
    y_vals = dist_func(x_vals)

    if np.max(y_vals) == 0:
        return

    y_max = 8

    axis_color = (180, 180, 180)
    tick_font = pygame.font.SysFont(None, 18)

    # -----------------------------
    # AXES
    # -----------------------------
    pygame.draw.line(screen, axis_color, (x0, y0 + h), (x0 + w, y0 + h), 1)
    pygame.draw.line(screen, axis_color, (x0, y0), (x0, y0 + h), 1)

    # -----------------------------
    # X TICKS (FIXED RANGE)
    # -----------------------------
    num_ticks = 8
    for i in range(num_ticks + 1):
        t = i / num_ticks
        x_val = x_min + t * (x_max - x_min)
        px = x0 + t * w

        pygame.draw.line(screen, axis_color, (px, y0 + h), (px, y0 + h + 6), 1)

        label = tick_font.render(f"{x_val:.2f}", True, (220, 220, 220))
        screen.blit(label, (px - 18, y0 + h + 8))

    # -----------------------------
    # Y TICKS (PDF SCALE)
    # -----------------------------
    for i in range(5):
        t = i / 4
        py = y0 + h - t * h

        pygame.draw.line(screen, axis_color, (x0 - 5, py), (x0, py), 1)

        label = tick_font.render(f"{t*y_max:.2f}", True, (220, 220, 220))
        screen.blit(label, (x0 - 55, py - 8))

    # -----------------------------
    # CURVE
    # -----------------------------
    points = []
    for i in range(len(x_vals)):
        px = x0 + (x_vals[i] - x_min) / (x_max - x_min) * w
        py = y0 + h - (y_vals[i] / y_max) * h
        points.append((px, py))

    pygame.draw.lines(screen, color, False, points, 2)

    # -----------------------------
    # TITLE
    # -----------------------------
    title_display = title + " (Normal)" if normal_mode else title
    title_txt = font.render(title_display, True, color)
    screen.blit(title_txt, (x0 + 5, y0 + 5))

    # -----------------------------
    # PARAM DISPLAY
    # -----------------------------
    if params is not None:
        mu, scale = params

        if normal_mode:
            info = font.render(f"μ={mu:.3f}  σ={scale:.3f}", True, (200, 200, 200))

        else:
            if title == "LT":
                info = font.render(f"μ={mu:.3f}  s={scale:.3f}", True, (200, 200, 200))

            elif title == "LLSA":
                info = font.render(f"μ={mu:.3f}  s={scale:.3f}", True, (200, 200, 200))

            elif title == "LLSB":
                info = font.render(f"μ={mu:.3f}  σ={scale:.3f}", True, (200, 200, 200))

            elif title == "CAM":
                info = font.render(f"ξ={mu:.3f}  ω={scale:.3f}", True, (200, 200, 200))

        screen.blit(info, (x0 + 5, y0 + 25))

    # -----------------------------
    # AXIS LABELS
    # -----------------------------
    y_label = tick_font.render("PDF", True, axis_color)
    screen.blit(y_label, (x0 - 55, y0 - 30))

    # Sensor-specific
    if params is not None:
        if title == "LT":
            x_label = "             Error in robot position"
        elif title == "CAM":
            x_label = "     Error in tow lateral movement"
        elif title == "LLSA":
            x_label = "Error in tow width before compaction"
        elif title == "LLSB":
            x_label = "Error in tow width after compaction"

        x_label = tick_font.render(x_label, True, axis_color)
        screen.blit(x_label, (x0 + w//2 - 100, y0 + h + 35))

def draw_all_distributions(screen, params):
    plot_w = 500
    plot_h = 350
    start_x = 800
    start_y = 130
    gap = 80

    draw_single_distribution(
        screen,
        NORMAL_distribution(*params["LT"]) if normal_mode else LT_distribution(*params["LT"]),
        (start_x, start_y, plot_w, plot_h),
        (0, 200, 255),
        "LT",
        params["LT"] if normal_mode else LT_parameters(*params["LT"])
    )

    draw_single_distribution(
        screen,
        NORMAL_distribution(*params["CAM"]) if normal_mode else CAM_distribution(*params["CAM"]),
        (start_x + plot_w + gap, start_y, plot_w, plot_h),
        (255, 200, 0),
        "CAM",
        params["CAM"] if normal_mode else CAM_parameters(*params["CAM"])
    )

    draw_single_distribution(
        screen,
        NORMAL_distribution(*params["LLSA"]) if normal_mode else LLSA_distribution(*params["LLSA"]),
        (start_x, start_y + plot_h + gap, plot_w, plot_h),
        (255, 100, 100),
        "LLSA",
        params["LLSA"] if normal_mode else LLSA_parameters(*params["LLSA"])
    )

    draw_single_distribution(
        screen,
        NORMAL_distribution(*params["LLSB"]) if normal_mode else LLSB_distribution(*params["LLSB"]),
        (start_x + plot_w + gap, start_y + plot_h + gap, plot_w, plot_h),
        (100, 255, 100),
        "LLSB",
        params["LLSB"] if normal_mode else LLSB_parameters(*params["LLSB"])
    )

def draw_exit_button():
    exit_button = pygame.Rect(WIDTH - 120, HEIGHT - 60, 100, 40)
    pygame.draw.rect(screen, (180, 50, 50), exit_button)
    txt = font.render("EXIT", True, (255, 255, 255))
    screen.blit(txt, (exit_button.x + 30, exit_button.y + 12))
    return exit_button

def draw_regenerate_button():
    regen_button = pygame.Rect(285, 450, 210, 40)

    pygame.draw.rect(screen, (70, 140, 220), regen_button)
    pygame.draw.rect(screen, (255, 255, 255), regen_button, 2)

    txt = font.render("REGENERATE TOWS", True, (255, 255, 255))
    screen.blit(txt, (regen_button.x + 23, regen_button.y + 13))

    return regen_button

def draw_reset_button():
    reset_button = pygame.Rect(285, 120, 210, 40)

    pygame.draw.rect(screen, (200, 140, 70), reset_button)
    pygame.draw.rect(screen, (255, 255, 255), reset_button, 2)

    txt = font.render("RESET VALUES", True, (255, 255, 255))
    screen.blit(txt, (reset_button.x + 45, reset_button.y + 13))

    return reset_button

def draw_normal_button():
    normal_button = pygame.Rect(285, 60, 210, 40)  # ABOVE reset

    pygame.draw.rect(screen, (100, 200, 120), normal_button)
    pygame.draw.rect(screen, (255, 255, 255), normal_button, 2)

    txt = font.render("NORMAL DIST", True, (255, 255, 255))
    screen.blit(txt, (normal_button.x + 50, normal_button.y + 13))

    return normal_button

def draw_multisim_button():
    multi_button = pygame.Rect(285, 500, 210, 40)  # below regenerate

    pygame.draw.rect(screen, (120, 80, 200), multi_button)
    pygame.draw.rect(screen, (255, 255, 255), multi_button, 2)

    txt = font.render("MULTI - SIMULATE", True, (255, 255, 255))
    screen.blit(txt, (multi_button.x + 35, multi_button.y + 13))

    return multi_button

def generate_tows_full_control(params, num_tows=3,
                              tow_spacing_mm=6.35,
                              tow_width_mm=6.35,
                              tow_length_mm=1000):

    tow_offset = 0
    top_paths, bottom_paths, centerlines = [], [], []

    LT_steps = STEP_CACHE["LT"]
    CAM_steps = STEP_CACHE["CAM"]
    LLSB_steps = STEP_CACHE["LLS_B"]
    LLSA_steps = STEP_CACHE["LLS_A"]

    for _ in range(num_tows):

        # --- generate each signal with YOUR parameters ---
        LT = generate_random_walk(
            "LT", LT_steps, RW_PROPOSAL_STD["LT"],
            NORMAL_distribution(*params["LT"]) if normal_mode else LT_distribution(*params["LT"]),
            RW_CACHE["LT"][3], RW_CACHE["LT"][4])

        CAM = generate_random_walk(
            "CAM", CAM_steps, RW_PROPOSAL_STD["CAM"],
            NORMAL_distribution(*params["CAM"]) if normal_mode else CAM_distribution(*params["CAM"]),
            RW_CACHE["CAM"][3], RW_CACHE["CAM"][4])

        LLSB = generate_random_walk(
            "LLS_B", LLSB_steps, RW_PROPOSAL_STD["LLS_B"],
            NORMAL_distribution(*params["LLSB"]) if normal_mode else LLSB_distribution(*params["LLSB"]),
            RW_CACHE["LLS_B"][3], RW_CACHE["LLS_B"][4])

        LLSA = generate_random_walk(
            "LLS_A", LLSA_steps, RW_PROPOSAL_STD["LLS_A"],
            NORMAL_distribution(*params["LLSA"]) if normal_mode else LLSA_distribution(*params["LLSA"]),
            RW_CACHE["LLS_A"][3], RW_CACHE["LLS_A"][4])

        # --- match lengths ---
        n_steps = min(len(LT), len(CAM), len(LLSB), len(LLSA))
        x = np.linspace(0, tow_length_mm, n_steps)

        def interp(arr):
            return np.interp(
                np.linspace(0, len(arr)-1, n_steps),
                np.arange(len(arr)),
                arr
            )

        LT = interp(LT)
        CAM = interp(CAM)
        LLSB = interp(LLSB)
        LLSA = interp(LLSA)

        # --- compaction logic ---
        compaction_error = -(LLSB - LLSA)
        compaction_error[compaction_error > 0] = 0

        # --- tow geometry ---
        center = tow_offset + CAM + LT
        width = tow_width_mm + LLSB

        top = center + 0.5 * width
        bottom = center - 0.5 * width

        top_paths.append(top)
        bottom_paths.append(bottom)
        centerlines.append(center)

        tow_offset += tow_spacing_mm

    # --- GAP / OVERLAP ---
    gap_area = 0
    overlap_area = 0

    for i in range(len(top_paths) - 1):
        diff = bottom_paths[i + 1] - top_paths[i]
        gap_area += np.trapezoid(np.clip(diff, 0, None), x)
        overlap_area += np.trapezoid(np.clip(-diff, 0, None), x)

    total_height = top_paths[-1] - bottom_paths[0]
    total_area = np.trapezoid(total_height, x)

    gap_percent = (gap_area / total_area) * 100 if total_area > 0 else 0
    overlap_percent = (overlap_area / total_area) * 100 if total_area > 0 else 0

    return x, top_paths, bottom_paths, centerlines, gap_percent, overlap_percent

def compute_gap_overlap_only(params, num_tows=31,
                             tow_spacing_mm=6.35,
                             tow_width_mm=6.35,
                             tow_length_mm=1000):

    LT_steps = STEP_CACHE["LT"]
    CAM_steps = STEP_CACHE["CAM"]
    LLSB_steps = STEP_CACHE["LLS_B"]
    LLSA_steps = STEP_CACHE["LLS_A"]

    tow_offset = 0

    top_paths = []
    bottom_paths = []

    # IMPORTANT: define x ONCE (same as full model behavior)
    x = np.linspace(0, tow_length_mm, min(LT_steps, CAM_steps, LLSB_steps, LLSA_steps))

    for _ in range(num_tows):

        LT = generate_random_walk(
            "LT", LT_steps, RW_PROPOSAL_STD["LT"],
            NORMAL_distribution(*params["LT"]) if normal_mode else LT_distribution(*params["LT"]),
            RW_CACHE["LT"][3], RW_CACHE["LT"][4]
        )

        CAM = generate_random_walk(
            "CAM", CAM_steps, RW_PROPOSAL_STD["CAM"],
            NORMAL_distribution(*params["CAM"]) if normal_mode else CAM_distribution(*params["CAM"]),
            RW_CACHE["CAM"][3], RW_CACHE["CAM"][4]
        )

        LLSB = generate_random_walk(
            "LLS_B", LLSB_steps, RW_PROPOSAL_STD["LLS_B"],
            NORMAL_distribution(*params["LLSB"]) if normal_mode else LLSB_distribution(*params["LLSB"]),
            RW_CACHE["LLS_B"][3], RW_CACHE["LLS_B"][4]
        )

        LLSA = generate_random_walk(
            "LLS_A", LLSA_steps, RW_PROPOSAL_STD["LLS_A"],
            NORMAL_distribution(*params["LLSA"]) if normal_mode else LLSA_distribution(*params["LLSA"]),
            RW_CACHE["LLS_A"][3], RW_CACHE["LLS_A"][4]
        )

        # interpolate EVERYTHING to SAME x (critical match to full model behavior)
        def interp(arr):
            return np.interp(
                np.linspace(0, len(arr)-1, len(x)),
                np.arange(len(arr)),
                arr
            )

        LT = interp(LT)
        CAM = interp(CAM)
        LLSB = interp(LLSB)
        LLSA = interp(LLSA)

        center = tow_offset + CAM + LT
        width = tow_width_mm + LLSB

        top = center + 0.5 * width
        bottom = center - 0.5 * width

        top_paths.append(top)
        bottom_paths.append(bottom)

        tow_offset += tow_spacing_mm

    # -----------------------------
    # GAP / OVERLAP (IDENTICAL LOGIC)
    # -----------------------------
    gap_area = 0
    overlap_area = 0

    for i in range(len(top_paths) - 1):
        diff = bottom_paths[i + 1] - top_paths[i]

        gap_area += np.trapezoid(np.clip(diff, 0, None), x)
        overlap_area += np.trapezoid(np.clip(-diff, 0, None), x)

    # IMPORTANT FIX: same normalization as full model
    total_height = top_paths[-1] - bottom_paths[0]
    total_area = np.trapezoid(total_height, x)

    gap_percent = (gap_area / total_area) * 100 if total_area > 0 else 0
    overlap_percent = (overlap_area / total_area) * 100 if total_area > 0 else 0

    return gap_percent, overlap_percent

def run_multi_simulation(params, runs=100, tows=31):
    gap_results = []
    overlap_results = []

    start_time = time.time()

    for _ in range(runs):
        gap, overlap = compute_gap_overlap_only(params, num_tows=tows)
        gap_results.append(gap)
        overlap_results.append(overlap)

    total_time = time.time() - start_time

    avg_gap = np.mean(gap_results)
    avg_overlap = np.mean(overlap_results)

    print("\n=== MULTI SIMULATION RESULTS ===")
    print(f"Runs: {runs}, Tows per run: {tows}")
    print(f"Average Gap: {avg_gap:.4f} %")
    print(f"Average Overlap: {avg_overlap:.4f} %")
    print(f"Total Time: {total_time:.2f} sec\n")

    return avg_gap, avg_overlap, total_time

def draw_tows(x, top_paths, bottom_paths, centerlines):
    origin_x = 100
    origin_y = 910
    width = 600
    height = 300

    # -----------------------------
    # WHITE PLOT BACKGROUND
    # -----------------------------
    plot_rect = pygame.Rect(origin_x, origin_y - height, width, height)
    pygame.draw.rect(screen, (255, 255, 255), plot_rect)

    x_min, x_max = min(x), max(x)

    all_y = np.concatenate(top_paths + bottom_paths)
    y_min, y_max = np.min(all_y), np.max(all_y)

    def transform(px, py):
        x_norm = (px - x_min) / (x_max - x_min)
        y_norm = (py - y_min) / (y_max - y_min)
        sx = origin_x + x_norm * width
        sy = origin_y - y_norm * height
        return int(sx), int(sy)

    # --- DRAW FILLED TOWS ---
    for i, (top, bottom, center) in enumerate(zip(top_paths, bottom_paths, centerlines)):

        pts_top = [transform(x[j], top[j]) for j in range(len(x))]
        pts_bottom = [transform(x[j], bottom[j]) for j in range(len(x))]

        polygon = pts_top + pts_bottom[::-1]

        color = (150, 150, 150) if i % 2 == 0 else (180, 180, 180)
        pygame.draw.polygon(screen, color, polygon)

        # edges
        edge_color = color  # same as fill
        pygame.draw.lines(screen, edge_color, False, pts_top, 1)
        pygame.draw.lines(screen, edge_color, False, pts_bottom, 1)

        # centerline
        pts_center = [transform(x[j], center[j]) for j in range(len(x))]
        pygame.draw.lines(screen, (255, 255, 255), False, pts_center, 1)

    # --- GAP / OVERLAP ---
    for i in range(len(top_paths) - 1):
        top_prev = top_paths[i]
        bottom_next = bottom_paths[i + 1]

        for j in range(len(x) - 1):
            diff = bottom_next[j] - top_prev[j]

            p1 = transform(x[j], top_prev[j])
            p2 = transform(x[j], bottom_next[j])
            p3 = transform(x[j+1], bottom_next[j+1])
            p4 = transform(x[j+1], top_prev[j+1])

            poly = [p1, p2, p3, p4]

            if diff > 0:
                pygame.draw.polygon(screen, (255, 255, 255), poly)  # GAP
            elif diff < 0:
                pygame.draw.polygon(screen, (114, 114, 114), poly)  # OVERLAP

        # -----------------------------
    # IDEAL CENTERLINES (DASHED GREY)
    # -----------------------------
    dash_color = (160, 160, 160)
    dash_length = 10
    gap_length = 8

    num_tows = len(top_paths)

    for i in range(num_tows):
        ideal_center = i * 6.35  # tow_spacing_mm

        # convert constant y-value line into screen space
        y = ideal_center

        # skip if outside view
        if y < y_min or y > y_max:
            continue

        # build dashed line across full width of plot
        x = x_min
        draw = True
        dist = 0

        while x < x_max:
            x2 = min(x + dash_length, x_max)

            if draw:
                p1 = transform(x, y)
                p2 = transform(x2, y)
                pygame.draw.line(screen, dash_color, p1, p2, 1)

            draw = not draw
            x += dash_length + gap_length
    
    # --- AXES ---
    axis_color = (220, 220, 220)
    tick_font = pygame.font.SysFont(None, 18)

    # X axis
    pygame.draw.line(screen, axis_color,
                    (origin_x, origin_y),
                    (origin_x + width, origin_y), 2)

    # X ticks every 200 mm
    for val in range(0, int(x_max)+1, 200):
        px, py = transform(val, y_min)

        pygame.draw.line(screen, axis_color, (px, origin_y), (px, origin_y + 6), 1)

        label = tick_font.render(f"{val}", True, axis_color)
        screen.blit(label, (px - 10, origin_y + 8))

    # Y axis
    pygame.draw.line(screen, axis_color,
                    (origin_x, origin_y - height),
                    (origin_x, origin_y), 2)

    # Y ticks every 2 mm
    y_tick_spacing = 2
    y_val = int(y_min)

    while y_val <= y_max:
        px, py = transform(x_min, y_val)

        pygame.draw.line(screen, axis_color, (origin_x - 6, py), (origin_x, py), 1)

        label = tick_font.render(f"{y_val}", True, axis_color)
        screen.blit(label, (origin_x - 50, py - 8))

        y_val += y_tick_spacing

    # Axis labels
    screen.blit(tick_font.render("Tow Length (mm)", True, axis_color),
                (origin_x + width // 2 - 60, origin_y + 35))

    screen.blit(tick_font.render("Position (mm)", True, axis_color),
                (origin_x - 80, origin_y - height - 30))

def draw_gap_and_overlap_percentages(gap_percent, overlap_percent):
    txt1 = font.render(f"Gap: {gap_percent:.2f} %", True, (255, 255, 255))
    txt2 = font.render(f"Overlap: {overlap_percent:.2f} %", True, (255, 255, 255))

    screen.blit(txt1, (280, 570))
    screen.blit(txt2, (380, 570))

def draw_loading_bar(progress, elapsed_time):
    bar_x, bar_y = 240, 300
    bar_w, bar_h = 300, 20

    # background
    pygame.draw.rect(screen, (80, 80, 80), (bar_x, bar_y, bar_w, bar_h))

    # progress fill
    pygame.draw.rect(screen, (100, 200, 255),
                     (bar_x, bar_y, int(bar_w * progress), bar_h))

    # border
    pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_w, bar_h), 2)

    # -----------------------------
    # TEXT: percentage
    # -----------------------------
    percent = progress * 100
    percent_txt = font.render(f"{percent:.1f} %", True, (255, 255, 255))
    screen.blit(percent_txt, (bar_x + bar_w // 2 - 20, bar_y + 3))

    # -----------------------------
    # TEXT: elapsed time
    # -----------------------------
    elapsed_txt = font.render(f"Elapsed: {elapsed_time:.1f}s", True, (255, 255, 255))
    screen.blit(elapsed_txt, (bar_x, bar_y + 30))

    # -----------------------------
    # TEXT: estimated remaining time
    # -----------------------------
    if progress > 0:
        total_estimated = elapsed_time / progress
        remaining = total_estimated - elapsed_time
    else:
        remaining = 0

    remaining_txt = font.render(f"Remaining: {remaining:.1f}s", True, (255, 255, 255))
    screen.blit(remaining_txt, (bar_x + 200, bar_y + 30))

############################################################################################################################################
"""Main loop"""

pygame.init()
WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

multi_running = False
multi_progress = 0
multi_start_time = 0
multi_total_runs = 10
multi_current_run = 0
multi_done_time = None
multi_gap_results = np.zeros(multi_total_runs)
multi_overlap_results = np.zeros(multi_total_runs)

class InputBox:
    def __init__(self, x, y, w, h, text, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = str(text)
        self.label = label
        self.active = False
        self.value = float(text)

    def handle_event(self, event):
        changed = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                try:
                    self.value = float(self.text)
                    changed = True
                except:
                    pass  # ignore invalid input
                self.active = False

            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]

            else:
                if event.unicode in "0123456789.-":
                    self.text += event.unicode

        return changed

    def draw(self, screen):
        color = (255, 100, 100) if self.active else (200, 200, 200)

        pygame.draw.rect(screen, (40, 40, 40), self.rect)
        pygame.draw.rect(screen, color, self.rect, 2)

        txt_surface = font.render(self.text, True, (255, 255, 255))
        screen.blit(txt_surface, (self.rect.x + 5, self.rect.y + 5))

        label_surface = font.render(self.label, True, (255, 255, 255))
        screen.blit(label_surface, (self.rect.x, self.rect.y - 20))

inputs = {
    "LT_shift": InputBox(285, 210, 100, 30, 0.0, "LT Shift"),
    "LT_scale": InputBox(395, 210, 100, 30, 1.0, "LT Scale"),

    "CAM_shift": InputBox(285, 270, 100, 30, 0.0, "CAM Shift"),
    "CAM_scale": InputBox(395, 270, 100, 30, 1.0, "CAM Scale"),

    "LLSB_mean": InputBox(285, 330, 100, 30, -0.08, "LLSB Mean"),
    "LLSB_std": InputBox(395, 330, 100, 30, 0.06, "LLSB STD"),

    "LLSA_shift": InputBox(285, 390, 100, 30, 0.0, "LLSA Shift"),
    "LLSA_scale": InputBox(395, 390, 100, 30, 1.0, "LLSA Scale"),
}

initial_input_values = {
    "LT_shift": (0.0, "0.0"),
    "LT_scale": (1.0, "1.0"),

    "CAM_shift": (0.0, "0.0"),
    "CAM_scale": (1.0, "1.0"),

    "LLSB_mean": (-0.08, "-0.08"),
    "LLSB_std": (0.06, "0.06"),

    "LLSA_shift": (0.0, "0.0"),
    "LLSA_scale": (1.0, "1.0")}

def update_input_labels(normal_mode):
    if normal_mode:
        inputs["LT_shift"].label = "LT μ"
        inputs["LT_scale"].label = "LT σ"

        inputs["CAM_shift"].label = "CAM μ"
        inputs["CAM_scale"].label = "CAM σ"

        inputs["LLSB_mean"].label = "LLSB μ"
        inputs["LLSB_std"].label = "LLSB σ"

        inputs["LLSA_shift"].label = "LLSA μ"
        inputs["LLSA_scale"].label = "LLSA σ"

    else:
        inputs["LT_shift"].label = "LT Shift"
        inputs["LT_scale"].label = "LT Scale"

        inputs["CAM_shift"].label = "CAM Shift"
        inputs["CAM_scale"].label = "CAM Scale"

        inputs["LLSB_mean"].label = "LLSB Mean"
        inputs["LLSB_std"].label = "LLSB STD"

        inputs["LLSA_shift"].label = "LLSA Shift"
        inputs["LLSA_scale"].label = "LLSA Scale"

# --- Initial tow values ---
x, top_paths, bottom_paths, centerlines, gap_percent, overlap_percent = generate_tows_full_control(params)
running = True
update_input_labels(normal_mode)

# store original params for reset
original_params = params.copy()
original_inputs = {k: (v.value, v.text) for k, v in inputs.items()}

while running:
    screen.fill((30, 30, 30))
    exit_button = draw_exit_button()
    regen_button = draw_regenerate_button()
    reset_button = draw_reset_button()
    normal_button = draw_normal_button()
    multi_button = draw_multisim_button()

    changed_any = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            # ALWAYS allow exit
            if exit_button.collidepoint(event.pos):
                running = False

            # Block other buttons during simulation
            if multi_running:
                continue

            if regen_button.collidepoint(event.pos):
                x, top_paths, bottom_paths, centerlines, gap_percent, overlap_percent = \
                    generate_tows_full_control(params)

            if reset_button.collidepoint(event.pos):
                normal_mode = False
                update_input_labels(normal_mode)

                for key, box in inputs.items():
                    val, text = original_inputs[key]
                    box.value = val
                    box.text = text

                params = original_params.copy()

                x, top_paths, bottom_paths, centerlines, gap_percent, overlap_percent = \
                    generate_tows_full_control(params)

            if normal_button.collidepoint(event.pos):
                normal_mode = True
                update_input_labels(normal_mode)

                # --- FORCE all to (-0.08, 0.06) ---
                default_mu = -0.08
                default_sigma = 0.06

                inputs["LT_shift"].value = default_mu
                inputs["LT_shift"].text = str(default_mu)
                inputs["LT_scale"].value = default_sigma
                inputs["LT_scale"].text = str(default_sigma)

                inputs["CAM_shift"].value = default_mu
                inputs["CAM_shift"].text = str(default_mu)
                inputs["CAM_scale"].value = default_sigma
                inputs["CAM_scale"].text = str(default_sigma)

                inputs["LLSB_mean"].value = default_mu
                inputs["LLSB_mean"].text = str(default_mu)
                inputs["LLSB_std"].value = default_sigma
                inputs["LLSB_std"].text = str(default_sigma)

                inputs["LLSA_shift"].value = default_mu
                inputs["LLSA_shift"].text = str(default_mu)
                inputs["LLSA_scale"].value = default_sigma
                inputs["LLSA_scale"].text = str(default_sigma)

                # update params immediately
                params = {
                    "LT": (default_mu, default_sigma),
                    "CAM": (default_mu, default_sigma),
                    "LLSB": (default_mu, default_sigma),
                    "LLSA": (default_mu, default_sigma)}

                x, top_paths, bottom_paths, centerlines, gap_percent, overlap_percent = \
                    generate_tows_full_control(params)

            if multi_button.collidepoint(event.pos):
                multi_running = True
                multi_progress = 0
                multi_current_run = 0

                multi_gap_results = np.zeros(multi_total_runs)
                multi_overlap_results = np.zeros(multi_total_runs)

                multi_start_time = time.time()

        # Disable typing during simulation (optional but cleaner)
        if not multi_running:
            for box in inputs.values():
                if box.handle_event(event):
                    changed_any = True
    
    if multi_running and running:
        # run a few iterations per frame (keeps UI responsive)
        batch_size = 1

        for _ in range(batch_size):
            if multi_current_run < multi_total_runs:
                gap, overlap = compute_gap_overlap_only(params, num_tows=31)

                multi_gap_results[multi_current_run] = gap
                multi_overlap_results[multi_current_run] = overlap

                multi_current_run += 1
            else:
                # finished
                multi_running = False

                avg_gap = np.mean(multi_gap_results)
                avg_overlap = np.mean(multi_overlap_results)
                total_time = time.time() - multi_start_time

                print("\n=== MULTI SIMULATION RESULTS ===")
                print(f"Average Gap: {avg_gap:.4f} %")
                print(f"Average Overlap: {avg_overlap:.4f} %")
                print(f"Total Time: {total_time:.2f} sec")

                break

        multi_progress = multi_current_run / multi_total_runs

    # ALWAYS update params (so plots update live)
    params = {
        "LT": (inputs["LT_shift"].value, inputs["LT_scale"].value),
        "CAM": (inputs["CAM_shift"].value, inputs["CAM_scale"].value),
        "LLSB": (inputs["LLSB_mean"].value, inputs["LLSB_std"].value),
        "LLSA": (inputs["LLSA_shift"].value, inputs["LLSA_scale"].value)}

    # ONLY regenerate tows when needed
    if changed_any:
        x, top_paths, bottom_paths, centerlines, gap_percent, overlap_percent = generate_tows_full_control(params)

    if multi_running:
        # ONLY draw loading UI
        elapsed = time.time() - multi_start_time
        draw_loading_bar(multi_progress, elapsed)

    else:
        # Normal UI rendering
        draw_tows(x, top_paths, bottom_paths, centerlines)

        for box in inputs.values():
            box.draw(screen)

        draw_all_distributions(screen, params)
        draw_gap_and_overlap_percentages(gap_percent, overlap_percent)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()

############################################################################################################################################
"""Paper Graphs"""

def compute_spacing_distribution(params, num_tows=31, tow_length_mm=1000):
    """
    Returns ALL local spacing values (mm) across all tow pairs and x positions.
    """

    LT_steps = STEP_CACHE["LT"]
    CAM_steps = STEP_CACHE["CAM"]
    LLSB_steps = STEP_CACHE["LLS_B"]
    LLSA_steps = STEP_CACHE["LLS_A"]

    tow_offset = 0
    top_paths = []
    bottom_paths = []

    x = np.linspace(0, tow_length_mm, min(LT_steps, CAM_steps, LLSB_steps, LLSA_steps))

    for _ in range(num_tows):

        LT = generate_random_walk(
            "LT", LT_steps, RW_PROPOSAL_STD["LT"],
            NORMAL_distribution(*params["LT"]),
            RW_CACHE["LT"][3], RW_CACHE["LT"][4]
        )

        CAM = generate_random_walk(
            "CAM", CAM_steps, RW_PROPOSAL_STD["CAM"],
            NORMAL_distribution(*params["CAM"]),
            RW_CACHE["CAM"][3], RW_CACHE["CAM"][4]
        )

        LLS_B = generate_random_walk(
            "LLS_B", LLSB_steps, RW_PROPOSAL_STD["LLS_B"],
            NORMAL_distribution(*params["LLSB"]),
            RW_CACHE["LLS_B"][3], RW_CACHE["LLS_B"][4]
        )

        LLS_A = generate_random_walk(
            "LLS_A", LLSA_steps, RW_PROPOSAL_STD["LLS_A"],
            NORMAL_distribution(*params["LLSA"]),
            RW_CACHE["LLS_A"][3], RW_CACHE["LLS_A"][4]
        )

        def interp(arr):
            return np.interp(
                np.linspace(0, len(arr)-1, len(x)),
                np.arange(len(arr)),
                arr
            )

        LT = interp(LT)
        CAM = interp(CAM)
        LLS_B = interp(LLS_B)
        LLS_A = interp(LLS_A)

        center = tow_offset + CAM + LT
        width = 6.35 + LLS_B

        top = center + 0.5 * width
        bottom = center - 0.5 * width

        top_paths.append(top)
        bottom_paths.append(bottom)

        tow_offset += 6.35

    # -----------------------------
    # COLLECT SPACING (KEY PART)
    # -----------------------------
    spacing_data = []

    for i in range(len(top_paths) - 1):
        diff = bottom_paths[i + 1] - top_paths[i]
        spacing_data.extend(diff)   # keep ALL values (not integrated)

    return np.array(spacing_data)

def compute_baseline_spacing(runs=100, tows=31):
    """
    Compute baseline spacing distribution ONCE and store in DataFrame.
    """

    baseline_params = {
        "LT": (-0.08, 0.06),
        "CAM": (-0.08, 0.06),
        "LLSB": (-0.08, 0.06),
        "LLSA": (-0.08, 0.06)}

    global normal_mode
    normal_mode = True

    baseline_data = []

    # progress bar
    def print_progress(i, total, start_time):
        progress = (i + 1) / total
        bar_len = 30
        filled = int(bar_len * progress)

        bar = "█" * filled + "-" * (bar_len - filled)
        percent = progress * 100

        elapsed = time.time() - start_time
        remaining = (elapsed / progress - elapsed) if progress > 0 else 0

        sys.stdout.write(
            f"\rBaseline |{bar}| {percent:6.2f}% "
            f"Elapsed: {elapsed:5.1f}s "
            f"Remaining: {remaining:5.1f}s"
        )
        sys.stdout.flush()

    start_time = time.time()

    for i in range(runs):
        baseline_data.extend(compute_spacing_distribution(baseline_params, num_tows=tows))
        print_progress(i, runs, start_time)

    print("\nBaseline ready.\n")

    baseline_df = pd.DataFrame({
        "spacing": baseline_data,
        "type": "baseline"
    })

    return baseline_df

def KDE_spacing_from_normals(custom_params=None, 
                             runs=100, 
                             tows=31, 
                             bins=30, 
                             customdata_df=None, 
                             baseline_df="Normal Distribution Variations/baseline_spacing_data.csv", 
                             save_csv=True, 
                             sensor_type=None, 
                             distribution_parameter=None,
                             plot=False):

    global normal_mode
    normal_mode = True

    # -----------------------------
    # VALID SENSOR INPUTS (UPDATED)
    # -----------------------------
    valid_sensors = ["LT", "CAM", "LLSB", "LLSA",
                     "LT_CAM", "LLSB_LLSA", "ALL"]
    valid_params = ["mu", "sigma", "both"]

    # -----------------------------
    # HELPER: LOAD DF OR CSV
    # -----------------------------
    def load_df(data):
        if data is None:
            return None
        if isinstance(data, str):
            print(f"Loading data from CSV: {data}")
            return pd.read_csv(data)
        return data

    baseline_df = load_df(baseline_df)
    customdata_df = load_df(customdata_df)

    # -----------------------------
    # VALIDATE BASELINE
    # -----------------------------
    if baseline_df is None:
        raise ValueError("baseline_df must be provided.")

    # -----------------------------
    # AUTO-LOAD CUSTOM DATA (KEY FEATURE)
    # -----------------------------
    if customdata_df is None and custom_params is None:

        if sensor_type is None or distribution_parameter is None:
            raise ValueError(
                "Provide sensor_type and distribution_parameter to auto-load CSV."
            )

        if sensor_type not in valid_sensors:
            raise ValueError(f"sensor_type must be one of {valid_sensors}")

        if distribution_parameter not in valid_params:
            raise ValueError("distribution_parameter must be 'mu' or 'sigma'")

        filename = f"Normal Distribution Variations/{sensor_type}_shifted_{distribution_parameter}_spacing_data.csv"

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Could not find {filename}")

        print(f"Auto-loading custom data: {filename}")
        customdata_df = pd.read_csv(filename)

    # -----------------------------
    # GENERATE CUSTOM DATA IF NEEDED
    # -----------------------------
    if customdata_df is None:

        if custom_params is None:
            raise ValueError("Provide custom_params OR customdata_df.")

        if save_csv:
            if sensor_type not in valid_sensors:
                raise ValueError(f"sensor_type must be one of {valid_sensors}")
            if distribution_parameter not in valid_params:
                raise ValueError("distribution_parameter must be 'mu' or 'sigma'")

        custom_data = []
        start_time = time.time()

        def print_progress(i, total):
            progress = (i + 1) / total
            bar_len = 30
            filled = int(bar_len * progress)

            bar = "█" * filled + "-" * (bar_len - filled)
            percent = progress * 100

            elapsed = time.time() - start_time
            remaining = (elapsed / progress - elapsed) if progress > 0 else 0

            sys.stdout.write(
                f"\rCustom   |{bar}| {percent:6.2f}% "
                f"Elapsed: {elapsed:6.1f}s "
                f"Remaining: {remaining:6.1f}s"
            )
            sys.stdout.flush()

        for i in range(runs):
            custom_data.extend(
                compute_spacing_distribution(custom_params, num_tows=tows)
            )
            print_progress(i, runs)

        print("\nCustom data generation done.\n")

        customdata_df = pd.DataFrame({
            "spacing": custom_data
        })

        # -----------------------------
        # SAVE CSV WITH NAMING
        # -----------------------------
        if save_csv:
            base_name = f"Normal Distribution Variations/{sensor_type}_shifted_{distribution_parameter}_spacing_data"
            ext = ".csv"
            final_name = base_name + ext

            counter = 1
            while os.path.exists(final_name):
                final_name = f"{base_name}_{counter}{ext}"
                counter += 1

            customdata_df.to_csv(final_name, index=False)
            print(f"Saved custom data to: {final_name}")

    else:
        print("Using precomputed custom data...")

    # -----------------------------
    # EXTRACT DATA
    # -----------------------------
    custom_data = customdata_df["spacing"].values
    baseline_data = baseline_df["spacing"].values

    # -----------------------------
    # STATS
    # -----------------------------
    print("\n=== SPACING (mm) ===")
    print(f"Custom mean/std:   {np.mean(custom_data):.4f} / {np.std(custom_data):.4f}")
    print(f"Baseline mean/std: {np.mean(baseline_data):.4f} / {np.std(baseline_data):.4f}")

    # -----------------------------
    # PLOT
    # -----------------------------
    if plot:
        plt.figure(figsize=(10, 6))

        plt.hist(baseline_data, bins=bins, density=True,
                 alpha=0.6, label="Baseline", color="blue")

        plt.hist(custom_data, bins=bins, density=True,
                 alpha=0.6, label="Custom", color="orange")

        try:
            import seaborn as sns
            sns.kdeplot(baseline_data, color="blue", linewidth=2)
            sns.kdeplot(custom_data, color="orange", linewidth=2)
        except:
            pass

        plt.axvline(0, color="black", linestyle=":", label="Ideal (0 gap)")

        plt.xlabel("Gap (mm)")
        plt.ylabel("Density")
        plt.legend()

        plt.tight_layout()
        plt.show()

    return customdata_df, baseline_df

def run_spacing_multiple_simulations(runs=100, tows=31, baseline_df="Normal Distribution Variations/baseline_spacing_data.csv", plot=False):

    base = {
        "LT": (-0.08, 0.06),
        "CAM": (-0.08, 0.06),
        "LLSB": (-0.08, 0.06),
        "LLSA": (-0.08, 0.06)}

    def modify(params, sensors=None, change=None):
        new_params = params.copy()

        if sensors is None:
            sensors = ["LT", "CAM", "LLSB", "LLSA"]

        for s in sensors:
            mu, sigma = new_params[s]

            if change == "mean":
                mu = 0
            elif change == "std":
                sigma = 0.12
            elif change == "both":
                mu = 0
                sigma = 0.12

            new_params[s] = (mu, sigma)

        return new_params

    experiments = [
        ("LT", "sigma", ["LT"]),
        ("LT", "mu", ["LT"]),
        ("CAM", "sigma", ["CAM"]),
        ("CAM", "mu", ["CAM"]),
        ("LT_CAM", "sigma", ["LT", "CAM"]),
        ("LT_CAM", "mu", ["LT", "CAM"]),
        ("LLSB", "sigma", ["LLSB"]),
        ("LLSB", "mu", ["LLSB"]),
        ("LLSA", "sigma", ["LLSA"]),
        ("LLSA", "mu", ["LLSA"]),
        ("LLSB_LLSA", "sigma", ["LLSB", "LLSA"]),
        ("LLSB_LLSA", "mu", ["LLSB", "LLSA"]),
        ("ALL", "both", ["LT", "CAM", "LLSB", "LLSA"]),
    ]

    total_exp = len(experiments)
    start_time = time.time()

    def print_progress(i):
        progress = (i + 1) / total_exp
        bar_len = 30
        filled = int(bar_len * progress)

        bar = "█" * filled + "-" * (bar_len - filled)
        percent = progress * 100

        elapsed = time.time() - start_time
        remaining = (elapsed / progress - elapsed) if progress > 0 else 0

        sys.stdout.write(
            f"\rBatch    |{bar}| {percent:6.2f}% "
            f"Elapsed: {elapsed:6.1f}s "
            f"Remaining: {remaining:6.1f}s"
        )
        sys.stdout.flush()

    for i, (name, param_type, sensors) in enumerate(experiments):

        print(f"\n===== Running: {name} | {param_type} =====")

        if param_type == "mu":
            change = "mean"
        elif param_type == "sigma":
            change = "std"
        elif param_type == "both":
            change = "both"

        custom_params = modify(base, sensors=sensors, change=change)

        KDE_spacing_from_normals(
            custom_params=custom_params,
            runs=runs,
            tows=tows,
            baseline_df=baseline_df,
            save_csv=True,
            sensor_type=name,
            distribution_parameter=param_type,
            plot=plot)

        print_progress(i)

    print("\n\nAll simulations complete.\n")

def plot_all_spacing_variations(baseline_file="Normal Distribution Variations/baseline_spacing_data.csv", bins=100, figsize=(18, 18)):
    """
    4x4 grid plot with manual loading bar and baseline vs baseline first.
    """

    cases = [
        ("LT_std", "Normal Distribution Variations/LT_shifted_sigma_spacing_data.csv"),
        ("LT_mu", "Normal Distribution Variations/LT_shifted_mu_spacing_data.csv"),

        ("CAM_std", "Normal Distribution Variations/CAM_shifted_sigma_spacing_data.csv"),
        ("CAM_mu", "Normal Distribution Variations/CAM_shifted_mu_spacing_data.csv"),

        ("LT+CAM_std", "Normal Distribution Variations/LT_CAM_shifted_sigma_spacing_data.csv"),
        ("LT+CAM_mu", "Normal Distribution Variations/LT_CAM_shifted_mu_spacing_data.csv"),

        ("LLSB_std", "Normal Distribution Variations/LLSB_shifted_sigma_spacing_data.csv"),
        ("LLSB_mu", "Normal Distribution Variations/LLSB_shifted_mu_spacing_data.csv"),

        ("LLSA_std", "Normal Distribution Variations/LLSA_shifted_sigma_spacing_data.csv"),
        ("LLSA_mu", "Normal Distribution Variations/LLSA_shifted_mu_spacing_data.csv"),

        ("LLSB+LLSA_std", "Normal Distribution Variations/LLSB_LLSA_shifted_sigma_spacing_data.csv"),
        ("LLSB+LLSA_mu", "Normal Distribution Variations/LLSB_LLSA_shifted_mu_spacing_data.csv"),

        ("ALL_both", "Normal Distribution Variations/ALL_shifted_both_spacing_data.csv"),
    ]

    total_steps = len(cases) + 1  # +1 for baseline load
    start_time = time.time()

    # -----------------------------
    # PROGRESS BAR
    # -----------------------------
    def print_progress(step, total, label="Loading"):
        progress = step / total
        bar_len = 30
        filled = int(bar_len * progress)

        bar = "█" * filled + "-" * (bar_len - filled)
        percent = progress * 100

        elapsed = time.time() - start_time
        remaining = (elapsed / progress - elapsed) if progress > 0 else 0

        sys.stdout.write(
            f"\r{label} |{bar}| {percent:6.2f}% "
            f"Elapsed: {elapsed:6.1f}s "
            f"Remaining: {remaining:6.1f}s"
        )
        sys.stdout.flush()

    # -----------------------------
    # LOAD BASELINE
    # -----------------------------
    baseline = pd.read_csv(baseline_file)["spacing"].values
    print_progress(1, total_steps)

    # -----------------------------
    # FIGURE SETUP (4x4)
    # -----------------------------
    fig, axes = plt.subplots(4, 4, figsize=figsize)
    axes = axes.flatten()

    def plot_case(ax, data, title):
        ax.hist(baseline, bins=bins, density=True, alpha=0.4, label="Baseline")
        ax.hist(data, bins=bins, density=True, alpha=0.4, label="Custom")

        try:
            sns.kdeplot(baseline, ax=ax, linewidth=1.5)
            sns.kdeplot(data, ax=ax, linewidth=1.5)
        except:
            pass

        ax.axvline(0, linestyle=":", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Gap (mm)")
        ax.set_ylabel("Density")
        ax.tick_params(labelsize=8)

    # -----------------------------
    # BASELINE vs BASELINE FIRST
    # -----------------------------
    plot_case(axes[0], baseline, "Baseline vs Baseline")

    # -----------------------------
    # LOAD + PLOT CASES
    # -----------------------------
    for i, (name, file) in enumerate(cases, start=1):

        if not os.path.exists(file):
            print(f"\nMissing file: {file}")
            continue

        df = pd.read_csv(file)
        data = df["spacing"].values

        plot_case(axes[i], data, name)

        print_progress(i + 1, total_steps)

    print("\nLoading complete.\n")

    # -----------------------------
    # HIDE UNUSED PANELS
    # -----------------------------
    for j in range(len(cases) + 1, len(axes)):
        axes[j].axis("off")

    # -----------------------------
    # GLOBAL LEGEND
    # -----------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

"""Generate Graphs"""
params_test = {
    "LT": (-0.08, 0.06),
    "CAM": (-0.08, 0.06),
    "LLSB": (-0.08, 0.06),
    "LLSA": (-0.08, 0.06)}
# KDE_spacing_from_normals(params_test, runs=1, plot=True, sensor_type="LT", distribution_parameter="mu")

"""Load Graphs"""
KDE_spacing_from_normals(sensor_type="LT", distribution_parameter="mu", plot=True, bins=100)

"""Run Simulations To Make A Lot Of Data"""
# run_spacing_multiple_simulations()

"""Plot all cases of changes"""
# plot_all_spacing_variations(bins=100)

