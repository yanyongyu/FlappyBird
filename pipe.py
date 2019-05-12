#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the pipe object of the game.
Author: yanyongyu
"""

__author__ = "yanyongyu"
__all__ = ["get_pipe"]

import random

import pygame

from utils import getHitmask

UPIPE_IMAGE = [pygame.image.load("images/game/pipe_down.png"),
               pygame.image.load("images/game/pipe2_down.png")]
DPIPE_IMAGE = [pygame.image.load("images/game/pipe_up.png"),
               pygame.image.load("images/game/pipe2_up.png")]
RANDOM_RECT = None
PIPEGAPSIZE = 100
PIPE_INDEX = None


class UPipe(pygame.sprite.Sprite):
    def __init__(self, bg_size, location):
        global RANDOM_RECT
        pygame.sprite.Sprite.__init__(self)
        self.image = UPIPE_IMAGE[PIPE_INDEX].convert_alpha()
        self.mask = getHitmask(self.image)

        self.width, self.height = bg_size
        self.rect = self.image.get_rect()
        self.rect.top = RANDOM_RECT - self.rect.height
        if location:
            self.rect.left = location
        else:
            self.rect.left = self.width + 10

    def move(self):
        self.rect.left -= 4


class DPipe(pygame.sprite.Sprite):
    def __init__(self, bg_size, location):
        global RANDOM_RECT
        pygame.sprite.Sprite.__init__(self)
        self.image = DPIPE_IMAGE[PIPE_INDEX].convert_alpha()
        self.mask = getHitmask(self.image)

        self.width, self.height = bg_size
        self.rect = self.image.get_rect()
        self.rect.top = RANDOM_RECT + PIPEGAPSIZE
        if location:
            self.rect.left = location
        else:
            self.rect.left = self.width + 10

    def move(self):
        self.rect.left -= 4


def get_pipe(bg_size, landy, location=None):
    global RANDOM_RECT
    gapYs = [20, 30, 40, 50, 60, 70, 80, 90]
#    RANDOM_RECT = random.randint(0.2*landy, 0.8*landy - PIPEGAPSIZE)
    RANDOM_RECT = gapYs[random.randint(0, len(gapYs)-1)] + int(0.2*landy)
    return UPipe(bg_size, location), DPipe(bg_size, location)
