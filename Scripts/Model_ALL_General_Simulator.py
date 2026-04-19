# External imports
import numpy as np
from scipy.stats import logistic, norm, skewnorm
import pygame

#Internal imports
from Handling_ALL_Functions import get_data
from Model_ALL_RandomWalk import generate_random_walk

############################################################################################################################################
"""Help Functions"""

def LT_distribution(mean_shift, scale_factor):
    mu = -0.93 + mean_shift
    s = 0.036 * scale_factor
    return lambda x: logistic.pdf(x, loc=mu, scale=s)

def LT_params(mean_shift, scale_factor):
    mu = -0.93 + mean_shift
    s = 0.036 * scale_factor
    return mu, s

def LLSA_distribution(mean_shift, scale_factor):
    mu = -0.25 + mean_shift
    s = 0.041 * scale_factor
    return lambda x: logistic.pdf(x, loc=mu, scale=s)

def LLSA_params(mean_shift, scale_factor):
    mu = -0.25 + mean_shift
    s = 0.041 * scale_factor
    return mu, s

def LLSB_distribution(mean_shift, scale_factor):
    mu = mean_shift
    sigma = scale_factor
    return lambda x: norm.pdf(x, loc=mu, scale=sigma)

def LLSB_params(mean_shift, scale_factor):
    mu = mean_shift
    sigma = scale_factor
    return mu, sigma

def CAM_distribution(mean_shift, scale_factor):
    xi = 0.53 + mean_shift
    omega = 0.32 * scale_factor
    alpha = -2.29
    return lambda x: skewnorm.pdf(x, a=alpha, loc=xi, scale=omega)

def CAM_params(mean_shift, scale_factor):
    xi = 0.53 + mean_shift
    omega = 0.32 * scale_factor
    alpha = -2.29
    return xi, omega

def NORMAL_distribution(mu, sigma):
    return lambda x: norm.pdf(x, loc=mu, scale=sigma)

def get_n_steps(sensor):
    """This function gets the number of steps, which is the number of data points in a one meter tow"""
    data, weights = get_data(sensor, format='separated')

    lengths = []
    for i in range(len(data)):
        lengths.append(len(data[i][:]))

    return int(np.average(lengths))

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
        LT_params(*params["LT"])
    )

    draw_single_distribution(
        screen,
        NORMAL_distribution(*params["CAM"]) if normal_mode else CAM_distribution(*params["CAM"]),
        (start_x + plot_w + gap, start_y, plot_w, plot_h),
        (255, 200, 0),
        "CAM",
        CAM_params(*params["CAM"])
    )

    draw_single_distribution(
        screen,
        NORMAL_distribution(*params["LLSA"]) if normal_mode else LLSA_distribution(*params["LLSA"]),
        (start_x, start_y + plot_h + gap, plot_w, plot_h),
        (255, 100, 100),
        "LLSA",
        LLSA_params(*params["LLSA"])
    )

    draw_single_distribution(
        screen,
        NORMAL_distribution(*params["LLSB"]) if normal_mode else LLSB_distribution(*params["LLSB"]),
        (start_x + plot_w + gap, start_y + plot_h + gap, plot_w, plot_h),
        (100, 255, 100),
        "LLSB",
        LLSB_params(*params["LLSB"])
    )

def draw_exit_button():
    exit_button = pygame.Rect(WIDTH - 120, HEIGHT - 60, 100, 40)
    pygame.draw.rect(screen, (180, 50, 50), exit_button)
    txt = font.render("EXIT", True, (255, 255, 255))
    screen.blit(txt, (exit_button.x + 30, exit_button.y + 12))

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

def generate_tows_full_control(params, num_tows=3,
                              tow_spacing_mm=6.35,
                              tow_width_mm=6.35,
                              tow_length_mm=1000):

    tow_offset = 0
    top_paths, bottom_paths, centerlines = [], [], []

    # --- get step counts ONCE ---
    LT_steps = get_n_steps("LT")
    CAM_steps = get_n_steps("CAM")
    LLSB_steps = get_n_steps("LLS_B")
    LLSA_steps = get_n_steps("LLS_A")

    for _ in range(num_tows):

        # --- generate each signal with YOUR parameters ---
        LT = generate_random_walk(
            "LT", LT_steps, 0.05,
            NORMAL_distribution(*params["LT"]) if normal_mode else LT_distribution(*params["LT"]),
            norm,
            (0, 1)
        )

        CAM = generate_random_walk(
            "CAM", CAM_steps, 0.05,
            NORMAL_distribution(*params["CAM"]) if normal_mode else CAM_distribution(*params["CAM"]),
            norm,
            (0, 1)
        )

        LLSB = generate_random_walk(
            "LLS_B", LLSB_steps, params["LLSB"][1],
            NORMAL_distribution(*params["LLSB"]) if normal_mode else LLSB_distribution(*params["LLSB"]),
            norm,
            params["LLSB"]
        )

        LLSA = generate_random_walk(
            "LLS_A", LLSA_steps, 0.05,
            NORMAL_distribution(*params["LLSA"]) if normal_mode else LLSA_distribution(*params["LLSA"]),
            norm,
            (0, 1)
        )

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

def draw_metrics(gap_percent, overlap_percent):
    txt1 = font.render(f"Gap: {gap_percent:.2f} %", True, (255, 255, 255))
    txt2 = font.render(f"Overlap: {overlap_percent:.2f} %", True, (255, 255, 255))

    screen.blit(txt1, (280, 540))
    screen.blit(txt2, (380, 540))

############################################################################################################################################
"""Main loop"""

pygame.init()
WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)
normal_mode = False

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
params = {
    "LT": (-0.93, 0.06),
    "CAM": (0.29, 0.22),
    "LLSB": (-0.08, 0.06),
    "LLSA": (-0.25, 0.07)}
x, top_paths, bottom_paths, centerlines, gap_percent, overlap_percent = generate_tows_full_control(params)
running = True
update_input_labels(normal_mode)

# store original params for reset
original_params = params.copy()
original_inputs = {k: (v.value, v.text) for k, v in inputs.items()}

while running:
    screen.fill((30, 30, 30))
    exit_button = pygame.Rect(WIDTH - 120, HEIGHT - 60, 100, 40)
    regen_button = pygame.Rect(285, 450, 210, 40)
    reset_button = pygame.Rect(285, 120, 210, 40)
    normal_button = pygame.Rect(285, 60, 210, 40)

    changed_any = False

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            if exit_button.collidepoint(event.pos):
                running = False

            if regen_button.collidepoint(event.pos):
                x, top_paths, bottom_paths, centerlines, gap_percent, overlap_percent = generate_tows_full_control(params)

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

                # convert ALL sensors to normal using current input values
                params = {
                    "LT": (inputs["LT_shift"].value, inputs["LT_scale"].value),
                    "CAM": (inputs["CAM_shift"].value, inputs["CAM_scale"].value),
                    "LLSB": (inputs["LLSB_mean"].value, inputs["LLSB_std"].value),
                    "LLSA": (inputs["LLSA_shift"].value, inputs["LLSA_scale"].value)
                }

                x, top_paths, bottom_paths, centerlines, gap_percent, overlap_percent = \
                    generate_tows_full_control(params)

        for box in inputs.values():
            if box.handle_event(event):
                changed_any = True

    # ALWAYS update params (so plots update live)
    params = {
        "LT": (inputs["LT_shift"].value, inputs["LT_scale"].value),
        "CAM": (inputs["CAM_shift"].value, inputs["CAM_scale"].value),
        "LLSB": (inputs["LLSB_mean"].value, inputs["LLSB_std"].value),
        "LLSA": (inputs["LLSA_shift"].value, inputs["LLSA_scale"].value)}

    # ONLY regenerate tows when needed
    if changed_any:
        x, top_paths, bottom_paths, centerlines, gap_percent, overlap_percent = generate_tows_full_control(params)

    # Draw everything
    draw_tows(x, top_paths, bottom_paths, centerlines)

    for box in inputs.values():
        box.draw(screen)

    draw_exit_button()
    draw_all_distributions(screen, params)
    draw_metrics(gap_percent, overlap_percent)
    draw_regenerate_button()
    draw_reset_button()
    draw_normal_button()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()