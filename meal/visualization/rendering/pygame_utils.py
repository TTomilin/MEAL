import pygame
from pygame.locals import DOUBLEBUF, HWSURFACE, QUIT, RESIZABLE, VIDEORESIZE


def run_static_resizeable_window(surface, fps=30):
    """
    window that can be resized and closed using gui
    """
    pygame.init()
    clock = pygame.time.Clock()
    window = pygame.display.set_mode(
        surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE
    )
    window.blit(surface, (0, 0))
    pygame.display.flip()
    try:
        while True:
            pygame.event.pump()
            event = pygame.event.wait()
            if event.type == QUIT:
                pygame.display.quit()
                pygame.quit()
            elif event.type == VIDEORESIZE:
                window = pygame.display.set_mode(
                    event.dict["size"], HWSURFACE | DOUBLEBUF | RESIZABLE
                )
                window.blit(
                    pygame.transform.scale(surface, event.dict["size"]), (0, 0)
                )
                pygame.display.flip()
                clock.tick(fps)
    except:
        pygame.display.quit()
        pygame.quit()
        if event.type != QUIT:  # if user meant to quit error does not matter
            raise


def vstack_surfaces(surfaces, background_color=None):
    """
    stack surfaces vertically (on y axis)
    if surfaces have different width fill remaining area with background color
    """
    result_width = max(surface.get_width() for surface in surfaces)
    result_height = sum(surface.get_height() for surface in surfaces)
    result_surface = pygame.surface.Surface((result_width, result_height))
    if background_color:
        result_surface.fill(background_color)
    next_surface_y_position = 0
    for surface in surfaces:
        result_surface.blit(surface, (0, next_surface_y_position))
        next_surface_y_position += surface.get_height()
    return result_surface


def scale_surface_by_factor(surface, scale_by_factor):
    """return scaled input surface (with size multiplied by scale_by_factor param)
    scales also content of the surface
    """
    unscaled_size = surface.get_size()
    scaled_size = tuple(int(dim * scale_by_factor) for dim in unscaled_size)
    return pygame.transform.scale(surface, scaled_size)


def blit_on_new_surface_of_size(surface, size, background_color=None):
    """blit surface on new surface of given size of surface (with no resize of its content), filling not covered parts of result area with background color"""
    result_surface = pygame.surface.Surface(size)
    if background_color:
        result_surface.fill(background_color)
    result_surface.blit(surface, (0, 0))
    return result_surface
