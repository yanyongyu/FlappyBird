#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the land object of the game.
Author: yanyongyu
"""

__author__ = "yanyongyu"
__all__ = ["Land"]

import pygame


class Land(pygame.sprite.Sprite):
    def __init__(self, bg_size):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("images/land.png").convert()
        self.mask = pygame.mask.from_surface(self.image)

        self.width, self.height = bg_size
        self.rect = self.image.get_rect()
        self.rect.top = self.height - self.rect.height
        self.LANDSHIFT = self.rect.width - self.width
        self.rect.left = 0

    def move(self):
        self.rect.left = -((-self.rect.left + 4) % self.LANDSHIFT)
