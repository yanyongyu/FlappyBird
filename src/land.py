#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the land object of the game.
@Author: yanyongyu
"""
__author__ = "yanyongyu"
__all__ = ["Land"]

import pygame

from utils import getHitmask


class Land(pygame.sprite.Sprite):

    def __init__(self, bg_size):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("assets/images/land.png").convert()
        self.mask = getHitmask(self.image)

        self.width, self.height = bg_size
        self.rect = self.image.get_rect()
        self.rect.top = int(self.height * 0.79)
        self.LANDSHIFT = self.rect.width - self.width
        self.rect.left = 0

    def move(self):
        self.rect.left = -((-self.rect.left + 4) % self.LANDSHIFT)
