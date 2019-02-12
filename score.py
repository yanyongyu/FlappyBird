#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the score/sql module of the game.
Author: yanyongyu
"""

__author__ = "yanyongyu"
__all__ = ["display", "show_score", "show_best", "Sql"]

import os
import time

import sqlite3
import pygame

# 游戏中成绩数字
display_images = [pygame.image.load("images/game/font0.png"),
                pygame.image.load("images/game/font1.png"),
                pygame.image.load("images/game/font2.png"),
                pygame.image.load("images/game/font3.png"),
                pygame.image.load("images/game/font4.png"),
                pygame.image.load("images/game/font5.png"),
                pygame.image.load("images/game/font6.png"),
                pygame.image.load("images/game/font7.png"),
                pygame.image.load("images/game/font8.png"),
                pygame.image.load("images/game/font9.png")]

# 结算面板成绩数字
score_images = [pygame.image.load("images/end/score0.png"),
                pygame.image.load("images/end/score1.png"),
                pygame.image.load("images/end/score2.png"),
                pygame.image.load("images/end/score3.png"),
                pygame.image.load("images/end/score4.png"),
                pygame.image.load("images/end/score5.png"),
                pygame.image.load("images/end/score6.png"),
                pygame.image.load("images/end/score7.png"),
                pygame.image.load("images/end/score8.png"),
                pygame.image.load("images/end/score9.png")]

def display(screen, bg_size, score):
    """Display the score in the center of screen."""
    digits = list(map(int, str(score)))
    total_width = 0
    
    for digit in digits:
        total_width += display_images[digit].get_width()
    
    x = (bg_size[0] - total_width) // 2
    
    for digit in digits:
        screen.blit(display_images[digit], (x, bg_size[1]*0.1))
        x += display_images[digit].get_width()
        
def show_score(screen, bg_size, score):
    """Display the score on the score panel."""
    SCOREY = 157
    SCOREX_RIGHT = 235
    digits = list(map(int, str(score)))[::-1]
    for digit in digits:
        digit_x = SCOREX_RIGHT - score_images[digit].get_width()
        screen.blit(score_images[digit], (digit_x, SCOREY))
        SCOREX_RIGHT -= score_images[digit].get_width()
        
def show_best(screen, bg_size, score):
    """Display the best score on the score panel."""
    SCOREY = 198
    SCOREX_RIGHT = 235
    digits = list(map(int, str(score)))[::-1]
    for digit in digits:
        digit_x = SCOREX_RIGHT - score_images[digit].get_width()
        screen.blit(score_images[digit], (digit_x, SCOREY))
        SCOREX_RIGHT -= score_images[digit].get_width()
        
class Sql():
    """Connect with the sqlite3 database."""
    path = os.getcwd()
    
    @classmethod
    def __connect(cls):
        if "record.db" not in os.listdir(cls.path):
            conn = sqlite3.connect("record.db")
            cursor = conn.cursor()
            cursor.execute("create table record (time int primary key, score int)")
            cursor.close()
            conn.commit()
            conn.close()
        
        conn = sqlite3.connect("record.db")
        cursor = conn.cursor()
        return cursor, conn
    
    @classmethod
    def __close(cls, cursor, conn, commit=False):
        cursor.close()
        if commit:
            conn.commit()
        conn.close()
    
    @classmethod
    def get_score(cls):
        cursor, conn = cls.__connect()
        cursor.execute("select * from `record` order by `score` desc")
        value = cursor.fetchall()
        cls.__close(cursor, conn)
        return value
    
    @classmethod
    def set_score(cls,score):
        cursor, conn = cls.__connect()
        timestamp = round(time.time())
        value = cls.get_score()
        if len(value) < 3:
            cursor.execute("insert into record (time, score) values (%s, %s)" % (timestamp, score))
        else:
            if score >= value[2][1]:
                cursor.execute("update `record` set `time`=%s,`score`=%s where `time`=%s" % (timestamp, score, value[2][0]))
        cls.__close(cursor, conn, commit=True)
        
        #是否是新纪录
        if not value or score > value[0][1]:
            return True
        return False
