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

# Import tow generator
from Model_ALL_Simulation import generate_multitow_layout

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

def run_simulation():
    global simulation_result
    plt.clf()
    # Pass tow length directly, generator will compute steps
    gap_df, gap_only, overlap_only, gap_percent, overlap_percent = generate_multitow_layout(
        num_tows=num_tows,
        tow_spacing_mm=tow_spacing_mm,
        tow_width_mm=tow_width_mm,
        tow_length_mm=tow_length_mm,
        plot=True
    )
    fig = plt.gcf()
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

# ---------------- UI Drawing Functions ----------------

def draw_menu():
    screen.fill((30, 30, 30))
    button_labels = ["Simulation", "Settings", "Quit"]
    button_width, button_height, spacing = 200, 50, 20
    start_y = SCREEN_HEIGHT // 2 - (len(button_labels) * button_height + (len(button_labels)-1)*spacing)//2
    for i, label in enumerate(button_labels):
        rect = pygame.Rect(SCREEN_WIDTH//2 - button_width//2, start_y + i*(button_height+spacing), button_width, button_height)
        draw_button(label, rect)

def draw_settings():
    global field_rects
    screen.fill((40, 40, 40))
    settings = [
        ("Number of Tows", num_tows),
        ("Tow Width (mm)", tow_width_mm),
        ("Tow Length (mm)", tow_length_mm),
        ("Tow Spacing (mm)", tow_spacing_mm)
    ]
    field_rects = []
    for i, (label_text, value) in enumerate(settings):
        y = 50 + i * 60
        label = font.render(f"{label_text}:", True, (255, 255, 255))
        screen.blit(label, (50, y))
        rect = pygame.Rect(300, y, 200, 40)
        pygame.draw.rect(screen, (255, 255, 255), rect, 2)
        value_str = input_text if active_input_field == i else str(value)
        screen.blit(font.render(value_str, True, (255, 255, 255)), (rect.x+5, rect.y+5))
        field_rects.append(rect)
    draw_button("Back", pygame.Rect(50, SCREEN_HEIGHT-70, 100, 40))

def handle_settings_event(event):
    global num_tows, tow_width_mm, tow_length_mm, tow_spacing_mm
    global active_input_field, input_text, state

    if event.type == pygame.MOUSEBUTTONDOWN:
        back_rect = pygame.Rect(50, SCREEN_HEIGHT-70, 100, 40)
        if back_rect.collidepoint(event.pos):
            active_input_field = None
            input_text = ""
            state = MENU
            return
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
    progress = min(elapsed/loading_estimated_time, 0.95) if simulation_thread.is_alive() else 1.0
    remaining = max(0, loading_estimated_time - elapsed)
    screen.fill((20, 20, 20))
    bar_width, bar_height = SCREEN_WIDTH//2, 30
    bar_x, bar_y = (SCREEN_WIDTH - bar_width)//2, SCREEN_HEIGHT//2
    pygame.draw.rect(screen, (70,70,70), (bar_x, bar_y, bar_width, bar_height))
    pygame.draw.rect(screen, (100,200,100), (bar_x, bar_y, int(bar_width*progress), bar_height))
    screen.blit(font.render("Generating simulation...", True, (255,255,255)), (SCREEN_WIDTH//2 - 150, bar_y - 40))
    remaining_text = f"Estimated time remaining: {remaining:.1f} s"
    remaining_surface = font.render(remaining_text, True, (200,200,200))
    text_x = SCREEN_WIDTH // 2 - remaining_surface.get_width() // 2
    text_y = bar_y + bar_height + 30
    screen.blit(remaining_surface, (text_x, text_y))
    pygame.display.flip()
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
                    start_y = SCREEN_HEIGHT//2 - (3*button_height + 2*spacing)//2
                    labels = ["Simulation","Settings","Quit"]
                    for i, label in enumerate(labels):
                        rect = pygame.Rect(SCREEN_WIDTH//2 - button_width//2, start_y + i*(button_height+spacing), button_width, button_height)
                        if rect.collidepoint(event.pos):
                            if label=="Simulation":
                                simulation_result = None
                                loading_start_time = time.time()
                                loading_estimated_time = 0.08*num_tows + 0.006*tow_length_mm - 6
                                simulation_thread = threading.Thread(target=run_simulation)
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