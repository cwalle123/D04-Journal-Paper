"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     This file runs the user interace for simulating tows.    ║
║        Please be aware of the units for the inputs and       ║
║         remember to press Enter when inputing values!        ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
"""Written by: Clifton-John Walle"""






























import pygame
import threading
import sys
import time
import os
import tkinter as tk
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import tow generator
from Model_ALL_Simulation import generate_multitow_layout
from Model_ALL_RandomWalk import generate_RW_multitow

# ---------------- Screen Setup ----------------
root = tk.Tk()
root.withdraw()
SCREEN_WIDTH = root.winfo_screenwidth()
SCREEN_HEIGHT = root.winfo_screenheight()
root.destroy()

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Tow Simulation Interface")
font = pygame.font.SysFont(None, 36)
clock = pygame.time.Clock()

# ---------------- UI States ----------------
MENU = "menu"
SETTINGS = "settings"
LOADING = "loading"
SIMULATION = "simulation"
state = MENU

# ---------------- Default Simulation Settings ----------------
num_tows = 2
tow_width_mm = 6.35
tow_length_mm = 1000
tow_spacing_mm = 6.35
steps_per_mm = 340 / 1000  # ratio: 340 steps = 1000 mm
visualize_gaps_overlaps = False
fill_tows = False
visualize_centerline = True
show_gridlines = True

# ---------------- Runtime Variables ----------------
active_input_field = None
input_text = ""
simulation_thread = None
simulation_result = None  # (fig, gap_percent, overlap_percent)
loading_start_time = None
loading_estimated_time = None
figure_counter = 1
save_confirmation = False
save_time = None
save_duration = 500  # milliseconds

# ---------------- Helper Functions ----------------

def draw_button(text, rect, active=True, green=False):
    color = (70, 130, 180) if active else (100, 100, 100)
    if green:
        color = (11, 230, 44)
    pygame.draw.rect(screen, color, rect)
    label = font.render(text, True, (255, 255, 255))
    screen.blit(label, label.get_rect(center=rect.center))

def run_simulation(GO=False, fill=False, centerline=True, gridlines=True):
    global simulation_result
    plt.clf()

    # --- Generate RW multitow data ---
    gap_overlap_df, gap_df, overlap_df, gap_percent, overlap_percent, RW_all_tows_data = generate_RW_multitow(
        num_tows=num_tows,
        tow_spacing_mm=tow_spacing_mm,
        tow_width_mm=tow_width_mm,
        tow_length_mm=tow_length_mm,
        proposal_type="RWM",
        print_statement=False
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [(0.6, 0.6, 0.6), (0.7, 0.7, 0.7)]

    x_vals = RW_all_tows_data[0]["x_mm"]
    tow_indices = range(num_tows)

    for i in tow_indices:
        tow_df = RW_all_tows_data[i]
        color = colors[i % 2]
        x = tow_df["x_mm"]
        top_edge = tow_df["top_edge"]
        bottom_edge = tow_df["bottom_edge"]

        if fill:
            ax.fill_between(x, bottom_edge, top_edge, color=color, alpha=0.8, label=f"Tow {i+1}" if i == 0 else "")
        else:
            ax.plot(x, top_edge, color=color, lw=1.8)
            ax.plot(x, bottom_edge, color=color, lw=1.8)
            if centerline:
                ax.plot(x, tow_df["centerline"], color=color, lw=1.0, linestyle="--")

    # --- Gaps and overlaps ---
    if GO:
        for i in range(num_tows - 1):
            top_edge_prev = RW_all_tows_data[i]["top_edge"]
            bottom_edge_next = RW_all_tows_data[i + 1]["bottom_edge"]
            diff = bottom_edge_next - top_edge_prev
            ax.fill_between(x_vals, top_edge_prev, bottom_edge_next, where=(diff > 0),
                            color="white", alpha=0.3)
            ax.fill_between(x_vals, top_edge_prev, bottom_edge_next, where=(diff < 0),
                            color="black", alpha=0.3)

    # --- Ideal straight lines ---
    if centerline:
        x_min = min(df["x_mm"].min() for df in RW_all_tows_data)
        x_max = max(df["x_mm"].max() for df in RW_all_tows_data)
        for i in range(num_tows):
            ideal_y = i * tow_width_mm
            ax.plot([x_min, x_max], [ideal_y, ideal_y],
                    color="gray", linestyle=":", lw=1.2, alpha=0.8)

    # --- Axes and grid ---
    ax.set_xlabel("Tow Length (mm)", fontname="Times New Roman", fontsize=15)
    ax.set_ylabel("Position (mm)", fontname="Times New Roman", fontsize=15)
    if gridlines:
        ax.grid(True, linestyle="--", alpha=0.8)
    else:
        ax.grid(False)

    plt.tight_layout()
    simulation_result = (fig, gap_percent, overlap_percent)

def render_matplotlib_figure(fig):
    # Set figure size to match screen proportion
    max_width = int(SCREEN_WIDTH * 0.8)
    max_height = int(SCREEN_HEIGHT * 0.8)
    
    # Adjust DPI to fill screen nicely
    dpi = 100
    fig.set_size_inches(max_width / dpi, max_height / dpi)
    
    # Draw figure to buffer
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    width, height = canvas.get_width_height()
    image = pygame.image.frombuffer(buf, (width, height), "RGBA")

    # Scale to max 80% (just in case)
    aspect_ratio = width / height
    if width > max_width or height > max_height:
        if aspect_ratio > 1:
            new_width = max_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
        image = pygame.transform.smoothscale(image, (new_width, new_height))
    else:
        new_width, new_height = width, height

    return image, new_width, new_height

def draw_screen_border(color=(255, 255, 255), thickness=10):
    pygame.draw.rect(screen, color, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), thickness)

# ---------------- UI Drawing Functions ----------------

def draw_menu():
    screen.fill((30, 30, 30))

     # --- Title ---
    title_text = "Predictive Model of Gap and Overlap Defects in AFP Composites"
    title_font = pygame.font.SysFont(None, 48, bold=False)
    title_surface = title_font.render(title_text, True, (255, 255, 255))
    title_x = SCREEN_WIDTH // 2 - title_surface.get_width() // 2
    title_y = SCREEN_HEIGHT // 2 - 150  # adjust vertical position as desired
    screen.blit(title_surface, (title_x, title_y))

    button_labels = ["Simulation", "Settings", "Quit"]
    button_width, button_height, spacing = 200, 50, 20
    start_y = SCREEN_HEIGHT // 2 - (len(button_labels) * button_height + (len(button_labels)-1)*spacing)//2 + 50
    for i, label in enumerate(button_labels):
        rect = pygame.Rect(SCREEN_WIDTH//2 - button_width//2, start_y + i*(button_height+spacing), button_width, button_height)
        draw_button(label, rect)
    draw_screen_border()

def draw_settings():
    global field_rects
    screen.fill((40, 40, 40))

    start_x_label = SCREEN_WIDTH/2 - 240
    start_x_input = start_x_label + 250
    start_y = SCREEN_HEIGHT/2 - 300
    vertical_spacing = 70

    settings = [
        ("Number of Tows", num_tows),
        ("Tow Width (mm)", tow_width_mm),
        ("Tow Length (mm)", tow_length_mm),
        ("Tow Spacing (mm)", tow_spacing_mm)
    ]

    field_rects = []
    for i, (label_text, value) in enumerate(settings):
        y = start_y + i * vertical_spacing
        label = font.render(f"{label_text}:", True, (255, 255, 255))
        screen.blit(label, (start_x_label, y))
        rect = pygame.Rect(start_x_input, y, 200, 40)
        pygame.draw.rect(screen, (255, 255, 255), rect, 2)
        value_str = input_text if active_input_field == i else str(value)
        screen.blit(font.render(value_str, True, (255, 255, 255)), (rect.x + 5, rect.y + 5))
        field_rects.append(rect)

    reminder_text = font.render("Remember to press ENTER to confirm a value", True, (200, 200, 0))
    screen.blit(reminder_text, (start_x_label - 35, start_y + len(settings) * vertical_spacing + 30))

    toggle_y = start_y + len(settings) * vertical_spacing + 90
    toggle_spacing = 70
    button_width, button_height = 350, 50

    visualize_rect = pygame.Rect(SCREEN_WIDTH/2 - button_width/2, toggle_y, button_width, button_height)
    fill_rect = pygame.Rect(SCREEN_WIDTH/2 - button_width/2, toggle_y + toggle_spacing, button_width, button_height)
    centerline_rect = pygame.Rect(SCREEN_WIDTH/2 - button_width/2, toggle_y + 2*toggle_spacing, button_width, button_height)
    gridlines_rect = pygame.Rect(SCREEN_WIDTH/2 - button_width/2, toggle_y + 3*toggle_spacing, button_width, button_height)

    draw_button(f"Gaps and Overlaps: {'ON' if visualize_gaps_overlaps else 'OFF'}", visualize_rect, green=visualize_gaps_overlaps)
    draw_button(f"Fill Tows: {'ON' if fill_tows else 'OFF'}", fill_rect, green=fill_tows)
    draw_button(f"Centerlines: {'ON' if visualize_centerline else 'OFF'}", centerline_rect, green=visualize_centerline)
    draw_button(f"Gridlines: {'ON' if show_gridlines else 'OFF'}", gridlines_rect, green=show_gridlines)

    draw_button("Back", pygame.Rect(50, 50, 100, 40))
    draw_screen_border()
    return visualize_rect, fill_rect, centerline_rect, gridlines_rect

def handle_settings_event(event):
    global num_tows, tow_width_mm, tow_length_mm, tow_spacing_mm
    global active_input_field, input_text, state
    global visualize_gaps_overlaps, fill_tows, visualize_centerline, show_gridlines

    visualize_rect, fill_rect, centerline_rect, gridlines_rect = draw_settings()  # ensure we have latest button rects

    if event.type == pygame.MOUSEBUTTONDOWN:
        back_rect = pygame.Rect(50, 50, 100, 40)
        if back_rect.collidepoint(event.pos):
            active_input_field = None
            input_text = ""
            state = MENU
            return
        
        # --- Handle toggle buttons ---
        if visualize_rect.collidepoint(event.pos):
            visualize_gaps_overlaps = not visualize_gaps_overlaps
            return
        elif fill_rect.collidepoint(event.pos):
            fill_tows = not fill_tows
            return
        elif centerline_rect.collidepoint(event.pos):
            visualize_centerline = not visualize_centerline
            return
        elif gridlines_rect.collidepoint(event.pos):
            show_gridlines = not show_gridlines
            return
        
        # --- Handle numeric fields ---
        for i, rect in enumerate(field_rects):
            if rect.collidepoint(event.pos):
                active_input_field = i
                input_text = ""
                return
            
    elif event.type == pygame.KEYDOWN and active_input_field is not None:
        if event.key == pygame.K_RETURN:
            try:
                value = float(input_text)
                if active_input_field == 0:
                    if value <= 0 or not value.is_integer(): raise ValueError()
                    num_tows = int(value)
                elif active_input_field == 1:
                    if value <= 0: raise ValueError()
                    tow_width_mm = value
                elif active_input_field == 2:
                    if value <= 0 or not value.is_integer(): raise ValueError()
                    tow_length_mm = int(value)
                elif active_input_field == 3:
                    if value <= 0: raise ValueError()
                    tow_spacing_mm = value
            except ValueError:
                print("Invalid input!")
            active_input_field = None
            input_text = ""
        elif event.key == pygame.K_BACKSPACE:
            input_text = input_text[:-1]
        else:
            char = event.unicode
            if char.isdigit() or (char == "." and "." not in input_text):
                input_text += char

def draw_loading_bar():
    global loading_start_time, loading_estimated_time, simulation_thread
    elapsed = time.time() - loading_start_time
    progress = min(elapsed / loading_estimated_time, 0.95) if simulation_thread.is_alive() else 1.0
    remaining = max(0, loading_estimated_time - elapsed)

    # Determine main message
    if elapsed < 4:
        base_text = "Receiving input parameters"
    elif 4 <= elapsed < 9:
        base_text = "Loading model"
    else:
        base_text = "Generating tows"

    # Create a looping dot animation (... → .. → .)
    dot_cycle = int((elapsed * 1) % 3)  # changes roughly every second
    dots = "." * dot_cycle if dot_cycle > 0 else "..."
    loading_text = f"{base_text}{dots}"

    # --- Draw UI ---
    screen.fill((20, 20, 20))
    bar_width, bar_height = SCREEN_WIDTH // 2, 30
    bar_x, bar_y = (SCREEN_WIDTH - bar_width) // 2, SCREEN_HEIGHT // 2

    pygame.draw.rect(screen, (70, 70, 70), (bar_x, bar_y, bar_width, bar_height))
    pygame.draw.rect(screen, (100, 200, 100), (bar_x, bar_y, int(bar_width * progress), bar_height))

    # Loading text
    text_surface = font.render(loading_text, True, (255, 255, 255))
    screen.blit(text_surface, (SCREEN_WIDTH // 2 - text_surface.get_width() // 2, bar_y - 40))

    # Remaining time
    remaining_text = f"Estimated time remaining: {remaining:.1f} s"
    remaining_surface = font.render(remaining_text, True, (200, 200, 200))
    text_x = SCREEN_WIDTH // 2 - remaining_surface.get_width() // 2
    text_y = bar_y + bar_height + 30
    screen.blit(remaining_surface, (text_x, text_y))

    draw_screen_border()
    pygame.display.flip()

    # When simulation thread finishes
    if not simulation_thread.is_alive():
        time.sleep(0.5)
        return True
    return False

def draw_simulation_screen():
    global simulation_result, save_confirmation, save_time, figure_counter
    screen.fill((0,0,0))
    if simulation_result is None:
        loading_text = font.render("Waiting for simulation result...", True, (255, 255, 255))
        screen.blit(loading_text, (50, SCREEN_HEIGHT//2))
        pygame.display.flip()
        return SIMULATION

    fig, gap_percent, overlap_percent = simulation_result
    image, new_width, new_height = render_matplotlib_figure(fig)
    x, y = (SCREEN_WIDTH-new_width)//2, (SCREEN_HEIGHT-new_height)//2
    screen.blit(image, (x, y))
    info_text = f"Gap %: {gap_percent:.2f}    Overlap %: {overlap_percent:.2f}"
    info_surface = font.render(info_text, True, (255,255,255))
    screen.blit(info_surface, (SCREEN_WIDTH//2 - info_surface.get_width()//2, y+new_height+30))

    # Buttons
    back_rect = pygame.Rect(50, SCREEN_HEIGHT-70, 100, 40)
    save_rect = pygame.Rect(SCREEN_WIDTH-150, SCREEN_HEIGHT-70, 100, 40)
    draw_button("Back", back_rect)
    draw_button("Save", save_rect, green=save_confirmation)
    draw_screen_border()
    pygame.display.flip()

    # Handle save confirmation timing
    if save_confirmation and pygame.time.get_ticks() - save_time > save_duration:
        save_confirmation = False

    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if back_rect.collidepoint(event.pos):
                return MENU
            elif save_rect.collidepoint(event.pos) and simulation_result is not None:
                os.makedirs("Figures", exist_ok=True)
                filename = os.path.join("Figures", f"figure_{figure_counter}.png")
                fig.savefig(filename)
                print(f"Saved figure as {filename}")
                figure_counter += 1
                save_confirmation = True
                save_time = pygame.time.get_ticks()
    return SIMULATION

# ---------------- Main Loop ----------------

def main():
    global state, simulation_thread, loading_start_time, loading_estimated_time, simulation_result
    while True:
        if state == MENU:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    button_width, button_height, spacing = 200, 50, 20
                    start_y = SCREEN_HEIGHT//2 - (3*button_height + 2*spacing)//2 + 50
                    labels = ["Simulation","Settings","Quit"]
                    for i, label in enumerate(labels):
                        rect = pygame.Rect(SCREEN_WIDTH//2 - button_width//2, start_y + i*(button_height+spacing), button_width, button_height)
                        if rect.collidepoint(event.pos):
                            if label=="Simulation":
                                simulation_result = None
                                loading_start_time = time.time()
                                loading_estimated_time = 0.237*num_tows + 12.6
                                simulation_thread = threading.Thread(target=run_simulation, kwargs=dict(GO=visualize_gaps_overlaps, fill=fill_tows, centerline=visualize_centerline, gridlines=show_gridlines))
                                simulation_thread.start()
                                state = LOADING
                            elif label=="Settings":
                                state = SETTINGS
                            elif label=="Quit":
                                pygame.quit()
                                sys.exit()
            draw_menu()

        elif state == SETTINGS:
            for event in pygame.event.get():
                handle_settings_event(event)
            draw_settings()

        elif state == LOADING:
            finished = draw_loading_bar()
            if finished:
                state = SIMULATION

        elif state == SIMULATION:
            state = draw_simulation_screen()

        pygame.display.flip()
        clock.tick(30)

if __name__ == "__main__":
    main()