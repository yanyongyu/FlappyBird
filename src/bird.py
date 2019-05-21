#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the bird object of the game.
Author: yan
"""

__author__ = "yanyongyu"
__all__ = ["Bird"]

import random
from itertools import cycle

import pygame

from utils import getHitmask

image = (
    [pygame.image.load("assets/images/birds/bird0_0.png"),
     pygame.image.load("assets/images/birds/bird0_1.png"),
     pygame.image.load("assets/images/birds/bird0_2.png")],
    [pygame.image.load("assets/images/birds/bird1_0.png"),
     pygame.image.load("assets/images/birds/bird1_1.png"),
     pygame.image.load("assets/images/birds/bird1_2.png")],
    [pygame.image.load("assets/images/birds/bird2_0.png"),
     pygame.image.load("assets/images/birds/bird2_1.png"),
     pygame.image.load("assets/images/birds/bird2_2.png")]
)


class Bird(pygame.sprite.Sprite):
    def __init__(self, bg_size, landy, color, ai=False):
        pygame.sprite.Sprite.__init__(self)

        if 0 <= color <= 2:
            self.images = list(map(lambda x: x.convert_alpha(), image[color]))
        elif color == 3:
            self.images = list(map(lambda x: x.convert_alpha(),
                                   image[random.choice([0, 1, 2])]))
        elif color == 4:
            try:
                self.images = [
                    pygame.image.load("assets/images/birds/custom_0.png")
                    .convert_alpha(),
                    pygame.image.load("assets/images/birds/custom_1.png")
                    .convert_alpha(),
                    pygame.image.load("assets/images/birds/custom_2.png")
                    .convert_alpha()
                ]
            except Exception:
                self.images = list(map(lambda x: x.convert_alpha(),
                                       image[random.choice([0, 1, 2])]))
        self.image_index_cycle = cycle([0, 1, 2, 1])
        self.index = next(self.image_index_cycle)
        self.image = self.images[self.index]
        self.mask = getHitmask(self.image)

        self.width, self.height = bg_size
        self.landy = landy
        self.rect = self.images[0].get_rect()
        self.rect.left = int(self.width * 0.2)
        self.rect.top = (self.height - self.rect.height) // 2
        self.center = self.rect.centerx, self.rect.centery

        self.alive = True
        self.ai = ai

        # 当前速度(负值向上)
        if self.ai:
            self.speedy = 0
        else:
            self.speedy = -9
        # 最大向下速度
        self.max_speedy = 10
        # 向下加速度
        self.down_speedy_a = 1
        # 角速度
        self.angular_speed = 3

        # 旋转角度
        self.rotate = 45

        if not self.ai:
            self.image = pygame.transform.rotate(self.images[self.index],
                                                 self.rotate)
        else:
            self.image = self.images[self.index]

    def image_index(self, delay):
        if delay % 3 == 0:
            self.index = next(self.image_index_cycle)
        return self.index

    def fly(self):
        self.speedy = -9
        if not self.ai:
            self.rotate = 45

    def move(self, delay):
        if self.ai:
            self.image = self.images[self.image_index(delay)]
            self.mask = getHitmask(self.image)
            self.center = self.rect.centerx, self.rect.centery
            self.rect = self.image.get_rect()
            self.rect.centerx, self.rect.centery = self.center
        else:
            if self.rotate >= -90:
                self.rotate -= self.angular_speed
                self.image = pygame.transform.rotate(
                    self.images[self.image_index(delay)],
                    self.rotate
                )
                self.mask = getHitmask(self.image)
                self.center = self.rect.centerx, self.rect.centery
                self.rect = self.image.get_rect()
                self.rect.centerx, self.rect.centery = self.center
        if self.landy - self.rect.top - self.rect.height + 20 >= 0:
            self.rect.top += min(self.speedy,
                                 self.landy
                                 - self.rect.top
                                 - self.rect.height
                                 + 20)
        if self.speedy <= self.max_speedy:
            self.speedy += self.down_speedy_a
