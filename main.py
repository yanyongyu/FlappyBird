#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main program of the game.
Author: yanyongyu
"""

__author__ = "yanyongyu"
__all__ = ["Game"]

import sys
import time
import random
import traceback

import pygame
import pygame.locals as gloc

import bird
import land
import pipe
import score
import share
import setting


class Game():

    def __init__(self):
        pygame.init()
        self.bg_size = self.width, self.height = 288, 512
        self.screen = pygame.display.set_mode(self.bg_size)
        pygame.display.set_caption("Flappy Bird")
        icon = pygame.image.load("images/flappy.ico")
        pygame.display.set_icon(icon)

        pygame.mixer.init()
        pygame.mixer.set_num_channels(4)

        self.clock = pygame.time.Clock()

        self.init_sound()
        self.init_pics()
        self.init_vars()

    def init_sound(self):
        self.sound = {}
        self.sound_default = {}
        # 死亡声音
        self.sound['die_sound'] = pygame.mixer.Sound("sound/die.wav")
        self.sound_default['die_sound'] = 0.4

        # 撞击声音
        self.sound['hit_sound'] = pygame.mixer.Sound("sound/hit.wav")
        self.sound_default['hit_sound'] = 0.4

        # 得分声音
        self.sound['point_sound'] = pygame.mixer.Sound("sound/point.wav")
        self.sound_default['point_sound'] = 0.4

        # 拍翅膀声音
        self.sound['wing_sound'] = pygame.mixer.Sound("sound/wing.wav")
        self.sound_default['wing_sound'] = 0.8

    def init_pics(self):
        # 加载背景与地面
        self.bg_black = pygame.image.load("images/bg_black.png")\
            .convert_alpha()
        self.background_list = [
                pygame.image.load("images/bg_day.png").convert(),
                pygame.image.load("images/bg_night.png").convert()
                ]
        self.land = land.Land(self.bg_size)

        # 游戏开始画面
        # 游戏标题
        self.title = pygame.image.load("images/start/title.png")\
            .convert_alpha()
        self.title_rect = self.title.get_rect()
        self.title_rect.left = (self.width - self.title_rect.width) // 2
        self.title_rect.top = 80

        # 开始按钮
        self.start_image = pygame.image.load("images/start/start.png")\
            .convert_alpha()
        self.start_image_rect = self.start_image.get_rect()
        self.start_image_rect.left = (self.width
                                      - self.start_image_rect.width) // 2
        self.start_image_rect.top = 240

        # 排行榜按钮
        self.score_image = pygame.image.load("images/start/score.png")\
            .convert_alpha()
        self.score_image_rect = self.score_image.get_rect()
        self.score_image_rect.left = (self.width
                                      - self.score_image_rect.width) // 2
        self.score_image_rect.top = 310

        # 设置按钮
        self.setting_image = pygame.image.load("images/start/setting.png")\
            .convert_alpha()
        self.setting_image_rect = self.setting_image.get_rect()
        self.setting_image_rect.left = (self.width
                                        - self.setting_image_rect.width
                                        - 10)
        self.setting_image_rect.top = 10

        # 排行画面
        # 奖杯
        self.cups = [
            pygame.image.load("images/rank/gold_cup.png").convert_alpha(),
            pygame.image.load("images/rank/silver_cup.png").convert_alpha(),
            pygame.image.load("images/rank/brooze_cup.png").convert_alpha()
            ]
        self.cup_rects = [(50, 120), (50, 200), (50, 280)]

        # 字体
        self.rank_font = pygame.font.Font("font/hanyihaiyun.ttf", 24)

        # 设置画面
        # 设置面板
        self.board_image = pygame.image.load("images/board.png")\
            .convert_alpha()
        self.board_rect = self.board_image.get_rect()
        self.board_rect.top = 20
        self.board_rect.left = (self.width - self.board_rect.width) // 2

        # 左右箭头
        self.array_right = pygame.image.load("images/start/array.png")\
            .convert_alpha()
        self.array_left = pygame.transform.rotate(self.array_right, 180)

        # 设置字体
        self.setting_font = pygame.font.Font("font/hanyihaiyun.ttf", 16)
        # 小鸟设置
        self.random_text = self.setting_font.render("随机", True, (0, 0, 0))
        self.custom_text = self.setting_font.render("自定义", True, (0, 0, 0))

        # 随机小鸟设置
        self.random_bird = [
                pygame.image.load("images/birds/random_0.png").convert_alpha(),
                pygame.image.load("images/birds/random_1.png").convert_alpha(),
                pygame.image.load("images/birds/random_2.png").convert_alpha()]
        # 自定义小鸟设置
        self.body_text = self.setting_font.render("身体", True, (0, 0, 0))
        self.mouth_text = self.setting_font.render("嘴", True, (0, 0, 0))

        self.R_text = self.setting_font.render("R", True, (0, 0, 0))
        self.G_text = self.setting_font.render("G", True, (0, 0, 0))
        self.B_text = self.setting_font.render("B", True, (0, 0, 0))

        self.customize_bird = setting.Customize_bird()

        # 背景设置
        self.bg_text = self.setting_font.render("背景：", True, (0, 0, 0))
        self.bg_text_list = [self.setting_font.render("白天", True, (0, 0, 0)),
                             self.setting_font.render("夜晚", True, (0, 0, 0)),
                             self.random_text]

        # 音量设置
        self.volume_text = self.setting_font.render("音量：", True, (0, 0, 0))
        self.sound_text = self.setting_font.render("音效：", True, (0, 0, 0))

        # 游戏画面
        # 准备图片
        self.ready_image = pygame.image.load("images/game/ready.png")\
            .convert_alpha()
        self.ready_rect = self.ready_image.get_rect()
        self.ready_rect.left = (self.width - self.ready_rect.width) // 2
        self.ready_rect.top = self.height * 0.12

        # 点击开始图片
        self.press_start_image = pygame.image.load("images/game/tutorial.png")\
            .convert_alpha()
        self.press_start_rect = self.press_start_image.get_rect()
        self.press_start_rect.left = (self.width
                                      - self.press_start_rect.width) // 2
        self.press_start_rect.top = self.height * 0.5

        # 暂停按钮
        self.pause_image = pygame.image.load("images/game/pause.png")\
            .convert_alpha()
        self.pause_image_rect = self.pause_image.get_rect()
        self.pause_image_rect.left = (self.width
                                      - self.pause_image_rect.width
                                      - 10)
        self.pause_image_rect.top = 10

        # 继续按钮
        self.resume_image = pygame.image.load("images/game/resume.png")\
            .convert_alpha()
        self.resume_image_rect = self.resume_image.get_rect()
        self.resume_image_rect.left = (self.width
                                       - self.resume_image_rect.width
                                       - 10)
        self.resume_image_rect.top = 10

        # 分享画面
        # 复制到剪贴板
        self.copy_image = pygame.image.load("images/share/copy.png")\
            .convert_alpha()
        self.copy_rect = self.copy_image.get_rect()
        self.copy_rect.left = (self.width - self.copy_rect.width) // 2
        self.copy_rect.top = 110

        # 保存至本地
        self.save_image = pygame.image.load("images/share/save.png")\
            .convert_alpha()
        self.save_rect = self.save_image.get_rect()
        self.save_rect.left = (self.width - self.save_rect.width) // 2
        self.save_rect.top = 200

        # 使用邮件分享
        self.email_image = pygame.image.load("images/share/email.png")\
            .convert_alpha()
        self.email_rect = self.email_image.get_rect()
        self.email_rect.left = (self.width - self.email_rect.width) // 2
        self.email_rect.top = 290

        # 返回
        self.back_image = pygame.image.load("images/share/back.png")\
            .convert_alpha()
        self.back_rect = self.back_image.get_rect()
        self.back_rect.left = (self.width - self.back_rect.width) // 2
        self.back_rect.top = 380

        # 游戏结束画面
        # 游戏结束图片
        self.gameover_image = pygame.image.load("images/end/gameover.png")\
            .convert_alpha()
        self.gameover_image_rect = self.gameover_image.get_rect()
        self.gameover_image_rect.left = (self.width
                                         - self.gameover_image_rect.width) // 2
        self.gameover_image_rect.top = self.height * 0.12

        # 得分面版
        self.score_panel = pygame.image.load("images/end/score_panel.png")\
            .convert_alpha()
        self.score_panel_rect = self.score_panel.get_rect()
        self.score_panel_rect.left = (self.width
                                      - self.score_panel_rect.width) // 2
        self.score_panel_rect.top = self.height * 0.24

        # 奖牌图片
        self.white_medal = pygame.image.load("images/end/medal0.png")\
            .convert_alpha()
        self.gold_medal = pygame.image.load("images/end/medal1.png")\
            .convert_alpha()
        self.silver_medal = pygame.image.load("images/end/medal2.png")\
            .convert_alpha()
        self.brooze_medal = pygame.image.load("images/end/medal3.png")\
            .convert_alpha()
        self.medal_rect = (57, 165)

        # 新纪录图片
        self.new_image = pygame.image.load("images/end/new.png")\
            .convert_alpha()
        self.new_rect = self.new_image.get_rect()
        self.new_rect.left, self.new_rect.top = 150, 139

        # 再来一次图片
        self.retry_image = pygame.image.load("images/end/retry.png")\
            .convert_alpha()
        self.retry_rect = self.retry_image.get_rect()
        self.retry_rect.left = (self.width - self.retry_rect.width) // 2
        self.retry_rect.top = self.height * 0.5

        # 分享按钮
        self.share_image = pygame.image.load("images/end/share.png")\
            .convert_alpha()
        self.share_rect = self.share_image.get_rect()
        self.share_rect.left = (self.width - self.share_rect.width) // 2
        self.share_rect.top = self.retry_rect.top + 30

        # 主菜单按钮
        self.menu_image = pygame.image.load("images/end/menu.png")\
            .convert_alpha()
        self.menu_rect = self.menu_image.get_rect()
        self.menu_rect.left = (self.width - self.menu_rect.width) // 2
        self.menu_rect.top = self.retry_rect.top + 60

    def init_vars(self, ai=False):
        # 读取设置
        (self.bird_color,
         self.background_index,
         self.volume,
         self.sound_volume) = setting.read_config()

        # 设置音量
        for i in self.sound.keys():
            self.sound[i].set_volume(
                    self.sound_volume
                    * self.sound_default[i]
                    / 100
                    )

        # 游戏分数
        self.score = 0

        # 背景
        if self.background_index == 2:
            pipe.PIPE_INDEX = random.choice([0, 1])
        elif self.background_index in [0, 1]:
            pipe.PIPE_INDEX = self.background_index
        self.background = self.background_list[pipe.PIPE_INDEX]

        # 游戏开始画面
        self.start = True

        # 排行榜画面
        self.ranking = False
        self.value = None

        # 设置画面
        self.setting = False
        self.mouse_down = False
        self.R1_set = setting.Setting_line(
                self.screen,
                rect=(64, 199),
                lenth=40,
                point=0.5,
                color=(255, 0, 0),
                height=3
                )
        self.G1_set = setting.Setting_line(
                self.screen,
                rect=(125, 199),
                lenth=40,
                point=0.5,
                color=(0, 255, 0),
                height=3
                )
        self.B1_set = setting.Setting_line(
                self.screen,
                rect=(189, 199),
                lenth=40,
                point=0.5,
                color=(0, 0, 255),
                height=3
                )
        self.R2_set = setting.Setting_line(
                self.screen,
                rect=(64, 249),
                lenth=40,
                point=0.5,
                color=(255, 0, 0),
                height=3
                )
        self.G2_set = setting.Setting_line(
                self.screen,
                rect=(125, 249),
                lenth=40,
                point=0.5,
                color=(0, 255, 0),
                height=3
                )
        self.B2_set = setting.Setting_line(
                self.screen,
                rect=(189, 249),
                lenth=40,
                point=0.5,
                color=(0, 0, 255),
                height=3
                )
        self.volume_set = setting.Setting_line(
                self.screen,
                rect=(105, 358),
                lenth=110,
                point=self.volume / 100,
                color=(230, 100, 0)
                )
        self.sound_set = setting.Setting_line(
                self.screen,
                rect=(105, 408),
                lenth=110,
                point=self.sound_volume/100,
                color=(230, 100, 0)
                )

        # 游戏画面
        self.bird = bird.Bird(self.bg_size, self.land.rect.top,
                              self.bird_color, ai=ai)
        self.delay = 0
        self.paused = False
        self.pressed = False
        self.upperpipes = []
        self.lowerpipes = []
        self.pipe_group = pygame.sprite.Group()
        if not ai:
            upipe, dpipe = pipe.get_pipe(self.bg_size, self.land.rect.top,
                                         self.width + 200)
        else:
            upipe, dpipe = pipe.get_pipe(self.bg_size, self.land.rect.top,
                                         self.width)
        self.upperpipes.append(upipe)
        self.lowerpipes.append(dpipe)
        self.pipe_group.add(upipe, dpipe)
        if not ai:
            upipe, dpipe = pipe.get_pipe(self.bg_size, self.land.rect.top,
                                         1.5*self.width + 200)
        else:
            upipe, dpipe = pipe.get_pipe(self.bg_size, self.land.rect.top,
                                         1.5*self.width)
        self.upperpipes.append(upipe)
        self.lowerpipes.append(dpipe)
        self.pipe_group.add(upipe, dpipe)

        # 游戏结束画面
        self.recorded = False

        # 分享画面
        self.share = False

    def play(self):
        while True:
            for event in pygame.event.get():
                # 退出事件
                if event.type == gloc.QUIT:
                    pygame.quit()
                    sys.exit()

# ==================================键盘事件===================================
                elif event.type == gloc.KEYDOWN:
                    # 空格/上键
                    if event.key == gloc.K_SPACE or event.key == gloc.K_UP:
                        # 游戏界面，小鸟存活，未暂停
                        # ----> 游戏开始/小鸟拍翅膀
                        if (not self.start
                                and not self.ranking
                                and not self.setting
                                and not self.paused
                                and self.bird.alive):
                            self.pressed = True
                            # 限制小鸟高度
                            if self.bird.rect.top > -2 * self.bird.rect.height:
                                self.bird.fly()
                                self.sound['wing_sound'].play()

                    # P键/Esc键
                    elif event.key == gloc.K_p or event.key == gloc.K_ESCAPE:
                        # 游戏界面，小鸟存活，未暂停
                        # ----> 游戏暂停/开始
                        if (not self.start
                                and not self.ranking
                                and not self.setting
                                and self.pressed
                                and self.bird.alive):
                            self.paused = not self.paused

# ================================鼠标移动事件==================================
                elif event.type == gloc.MOUSEMOTION:
                    # 设置界面
                    if self.setting and self.mouse_down:
                        pos = pygame.mouse.get_pos()
                        # RGB设置
                        # 身体
                        if pygame.Rect(64, 195, 40, 11).collidepoint(pos):
                            self.body_rgb[0] = (pos[0]-64) * 255 / 40
                            self.R1_set.set_point(self.body_rgb[0]/255)
                            self.customize_bird.seperate(self.body_rgb,
                                                         self.mouth_rgb)
                            self.bird = bird.Bird(self.bg_size,
                                                  self.land.rect.top,
                                                  self.bird_color)
                        elif pygame.Rect(125, 195, 40, 11).collidepoint(pos):
                            self.body_rgb[1] = (pos[0]-125) * 255 / 40
                            self.G1_set.set_point(self.body_rgb[1]/255)
                            self.customize_bird.seperate(self.body_rgb,
                                                         self.mouth_rgb)
                            self.bird = bird.Bird(self.bg_size,
                                                  self.land.rect.top,
                                                  self.bird_color)
                        elif pygame.Rect(189, 195, 40, 11).collidepoint(pos):
                            self.body_rgb[2] = (pos[0]-189) * 255 / 40
                            self.B1_set.set_point(self.body_rgb[2]/255)
                            self.customize_bird.seperate(self.body_rgb,
                                                         self.mouth_rgb)
                            self.bird = bird.Bird(self.bg_size,
                                                  self.land.rect.top,
                                                  self.bird_color)

                        # 嘴
                        elif pygame.Rect(64, 245, 40, 11).collidepoint(pos):
                            self.mouth_rgb[0] = (pos[0]-64) * 255 / 40
                            self.R2_set.set_point(self.mouth_rgb[0]/255)
                            self.customize_bird.seperate(self.body_rgb,
                                                         self.mouth_rgb)
                            self.bird = bird.Bird(self.bg_size,
                                                  self.land.rect.top,
                                                  self.bird_color)
                        elif pygame.Rect(125, 245, 40, 11).collidepoint(pos):
                            self.mouth_rgb[1] = (pos[0]-125) * 255 / 40
                            self.G2_set.set_point(self.mouth_rgb[1]/255)
                            self.customize_bird.seperate(self.body_rgb,
                                                         self.mouth_rgb)
                            self.bird = bird.Bird(self.bg_size,
                                                  self.land.rect.top,
                                                  self.bird_color)
                        elif pygame.Rect(189, 245, 40, 11).collidepoint(pos):
                            self.mouth_rgb[2] = (pos[0]-189) * 255 / 40
                            self.B2_set.set_point(self.mouth_rgb[2]/255)
                            self.customize_bird.seperate(self.body_rgb,
                                                         self.mouth_rgb)
                            self.bird = bird.Bird(self.bg_size,
                                                  self.land.rect.top,
                                                  self.bird_color)

                        # 音量设置
                        elif pygame.Rect(105, 352, 110, 15).collidepoint(pos):
                            self.volume = (pos[0]-105) * 100 / 110
                            self.volume_set.set_point(self.volume / 100)
                        elif pygame.Rect(105, 402, 110, 15).collidepoint(pos):
                            self.sound_volume = (pos[0]-105) * 100 / 110
                            self.sound_set.set_point(self.sound_volume / 100)
                            for i in self.sound.keys():
                                self.sound[i].set_volume(
                                        self.sound_volume
                                        * self.sound_default[i]
                                        / 100)

                        # 移出区域视为设置结束
                        else:
                            self.mouse_down = False

# =============================================================================
# ================================鼠标点击释放==================================
                elif event.type == gloc.MOUSEBUTTONUP:
                    # 设置界面
                    if self.setting and self.mouse_down:
                        self.mouse_down = False

# =============================================================================
# ================================鼠标点击事件=================================
                elif event.type == gloc.MOUSEBUTTONDOWN:
                    pos = event.pos
                    # 鼠标左键
                    if event.button == 1:
                        # 开始界面
                        if self.start:
                            # 进入游戏界面
                            if self.start_image_rect.collidepoint(pos):
                                self.start = False
                            # 进入排行界面
                            elif self.score_image_rect.collidepoint(pos):
                                self.start = False
                                self.ranking = True
                            # 进入设置界面
                            elif self.setting_image_rect.collidepoint(pos):
                                self.start = False
                                self.setting = True

                        # 排行榜界面
                        elif self.ranking:
                            # 回到开始界面
                            if self.back_rect.collidepoint(pos):
                                self.ranking = False
                                self.start = True

                        # 设置界面
                        elif self.setting:
                            # 回到开始界面
                            if self.setting_image_rect.collidepoint(pos):
                                self.start = True
                                self.setting = False
                                setting.write_json(
                                        self.bird_color,
                                        self.background_index,
                                        self.volume,
                                        self.sound_volume
                                        )

                            # 小鸟设置
                            elif pygame.Rect(52, 105, 30, 30)\
                                    .collidepoint(pos):
                                self.bird_color = (self.bird_color - 1) % 5
                                self.bird = bird.Bird(self.bg_size,
                                                      self.land.rect.top,
                                                      self.bird_color)
                            elif pygame.Rect(202, 105, 30, 30)\
                                    .collidepoint(pos):
                                self.bird_color = (self.bird_color + 1) % 5
                                self.bird = bird.Bird(self.bg_size,
                                                      self.land.rect.top,
                                                      self.bird_color)

                            # RGB设置
                            # 身体
                            elif pygame.Rect(64, 195, 40, 11)\
                                    .collidepoint(pos):
                                self.mouse_down = True
                                self.body_rgb[0] = (pos[0]-64) * 255 / 40
                                self.R1_set.set_point(self.body_rgb[0]/255)
                                self.customize_bird.seperate(self.body_rgb,
                                                             self.mouth_rgb)
                                self.bird = bird.Bird(self.bg_size,
                                                      self.land.rect.top,
                                                      self.bird_color)
                            elif pygame.Rect(125, 195, 40, 11)\
                                    .collidepoint(pos):
                                self.mouse_down = True
                                self.body_rgb[1] = (pos[0]-125) * 255 / 40
                                self.G1_set.set_point(self.body_rgb[1]/255)
                                self.customize_bird.seperate(self.body_rgb,
                                                             self.mouth_rgb)
                                self.bird = bird.Bird(self.bg_size,
                                                      self.land.rect.top,
                                                      self.bird_color)
                            elif pygame.Rect(189, 195, 40, 11)\
                                    .collidepoint(pos):
                                self.mouse_down = True
                                self.body_rgb[2] = (pos[0]-189) * 255 / 40
                                self.B1_set.set_point(self.body_rgb[2]/255)
                                self.customize_bird.seperate(self.body_rgb,
                                                             self.mouth_rgb)
                                self.bird = bird.Bird(self.bg_size,
                                                      self.land.rect.top,
                                                      self.bird_color)

                            # 嘴
                            elif pygame.Rect(64, 245, 40, 11)\
                                    .collidepoint(pos):
                                self.mouse_down = True
                                self.mouth_rgb[0] = (pos[0]-64) * 255 / 40
                                self.R2_set.set_point(self.mouth_rgb[0]/255)
                                self.customize_bird.seperate(self.body_rgb,
                                                             self.mouth_rgb)
                                self.bird = bird.Bird(self.bg_size,
                                                      self.land.rect.top,
                                                      self.bird_color)
                            elif pygame.Rect(125, 245, 40, 11)\
                                    .collidepoint(pos):
                                self.mouse_down = True
                                self.mouth_rgb[1] = (pos[0]-125) * 255 / 40
                                self.G2_set.set_point(self.mouth_rgb[1]/255)
                                self.customize_bird.seperate(self.body_rgb,
                                                             self.mouth_rgb)
                                self.bird = bird.Bird(self.bg_size,
                                                      self.land.rect.top,
                                                      self.bird_color)
                            elif pygame.Rect(189, 245, 40, 11)\
                                    .collidepoint(pos):
                                self.mouse_down = True
                                self.mouth_rgb[2] = (pos[0]-189) * 255 / 40
                                self.B2_set.set_point(self.mouth_rgb[2]/255)
                                self.customize_bird.seperate(self.body_rgb,
                                                             self.mouth_rgb)
                                self.bird = bird.Bird(self.bg_size,
                                                      self.land.rect.top,
                                                      self.bird_color)

                            # 背景设置
                            elif pygame.Rect(100, 292, 30, 30)\
                                    .collidepoint(pos):
                                self.background_index = (self.background_index
                                                         - 1) % 3
                                if self.background_index != 2:
                                    self.background = self.background_list[
                                            self.background_index
                                            ]
                            elif pygame.Rect(200, 292, 30, 30)\
                                    .collidepoint(pos):
                                self.background_index = (self.background_index
                                                         + 1) % 3
                                if self.background_index != 2:
                                    self.background = self.background_list[
                                            self.background_index
                                            ]

                            # 音量设置
                            elif pygame.Rect(105, 352, 110, 15)\
                                    .collidepoint(pos):
                                self.mouse_down = True
                                self.volume = (pos[0]-105) * 100 / 110
                                self.volume_set.set_point(self.volume / 100)
                            elif pygame.Rect(105, 402, 110, 15)\
                                    .collidepoint(pos):
                                self.mouse_down = True
                                self.sound_volume = (pos[0]-105) * 100 / 110
                                self.sound_set.set_point(self.sound_volume
                                                         / 100)

                        # 分享画面
                        elif self.share:
                            if self.copy_rect.collidepoint(pos):
                                try:
                                    share.copy(self.image_data)
                                except AttributeError:
                                    pass
                            elif self.save_rect.collidepoint(pos):
                                share.save(self.image_data)
                            elif self.email_rect.collidepoint(pos):
                                share.Email(self.image_data, self.score)
                            elif self.back_rect.collidepoint(pos):
                                self.share = False

                        # 游戏界面，小鸟存活
                        elif (self.pressed
                              and self.bird.alive
                              and self.pause_image_rect.collidepoint(pos)):
                            self.paused = not self.paused

                        # ----> 游戏开始/小鸟拍翅膀
                        elif not self.paused and self.bird.alive:
                            self.pressed = True
                            # 限制小鸟高度
                            if self.bird.rect.top > -2 * self.bird.rect.height:
                                self.bird.fly()
                                self.sound['wing_sound'].play()

                        # 游戏结束界面
                        elif not self.bird.alive:
                            pos = pygame.mouse.get_pos()
                            if self.retry_rect.collidepoint(pos):
                                self.init_vars()
                                self.start = False
                            elif self.share_rect.collidepoint(pos):
                                self.image_data = pygame.surfarray.array3d(
                                        pygame.display.get_surface())
                                self.share = True
                            elif self.menu_rect.collidepoint(pos):
                                self.init_vars()

            self.screen.blit(self.background, (0, 0))
            # 绘制地面
            self.screen.blit(self.land.image, self.land.rect)
            if self.bird.alive and not self.paused:
                self.land.move()
# ===============================游戏开始画面==================================
            if self.start:
                # 绘制游戏名
                self.screen.blit(self.title, self.title_rect)
                # 绘制开始按钮
                self.screen.blit(self.start_image, self.start_image_rect)
                # 绘制排行按钮
                self.screen.blit(self.score_image, self.score_image_rect)
                # 绘制设置按钮
                self.screen.blit(self.setting_image, self.setting_image_rect)

# ===========================================================================
# ==================================设置======================================
            elif self.setting:
                self.screen.blit(self.board_image, self.board_rect)
                self.screen.blit(self.setting_image, self.setting_image_rect)

                # 绘制小鸟设置
                self.screen.blit(self.array_left, (52, 105))
                self.screen.blit(self.array_right, (202, 105))
                if self.bird_color in [0, 1, 2]:
                    self.screen.blit(
                        self.bird.images[self.bird.image_index(self.delay)],
                        (120, 100))
                elif self.bird_color == 3:
                    self.screen.blit(
                        self.random_bird[self.bird.image_index(self.delay)],
                        (120, 100))
                    self.screen.blit(
                        self.random_text,
                        ((self.width-self.random_text.get_width()) // 2, 150))
                elif self.bird_color == 4:
                    self.screen.blit(
                        self.bird.images[self.bird.image_index(self.delay)],
                        (120, 100))
                    self.screen.blit(
                        self.custom_text,
                        ((self.width-self.custom_text.get_width()) // 2, 150))
                    self.screen.blit(
                        self.body_text,
                        ((self.width-self.body_text.get_width()) // 2, 170))
                    self.body_rgb = list(self.bird.images[0].get_at((23, 24)))
                    self.screen.blit(self.R_text, (50, 190))
                    self.R1_set.set_point(self.body_rgb[0] / 255)
                    self.R1_set.display()
                    self.screen.blit(self.G_text, (113, 190))
                    self.G1_set.set_point(self.body_rgb[1] / 255)
                    self.G1_set.display()
                    self.screen.blit(self.B_text, (175, 190))
                    self.B1_set.set_point(self.body_rgb[2] / 255)
                    self.B1_set.display()
                    self.screen.blit(
                        self.mouth_text,
                        ((self.width-self.mouth_text.get_width()) // 2, 220))
                    self.mouth_rgb = list(self.bird.images[0].get_at((30, 27)))
                    self.screen.blit(self.R_text, (50, 240))
                    self.R2_set.set_point(self.mouth_rgb[0] / 255)
                    self.R2_set.display()
                    self.screen.blit(self.G_text, (113, 240))
                    self.G2_set.set_point(self.mouth_rgb[1] / 255)
                    self.G2_set.display()
                    self.screen.blit(self.B_text, (175, 240))
                    self.B2_set.set_point(self.mouth_rgb[2] / 255)
                    self.B2_set.display()

                # 绘制背景设置
                self.screen.blit(self.bg_text, (50, 300))
                self.screen.blit(self.array_left, (100, 292))
                self.screen.blit(self.array_right, (200, 292))
                self.screen.blit(self.bg_text_list[self.background_index],
                                 (150, 300))

                # 绘制音量设置
                self.screen.blit(self.volume_text, (50, 350))
                self.volume_set.display()

                # 绘制音效设置
                self.screen.blit(self.sound_text, (50, 400))
                self.sound_set.display()

# =================================排行界面====================================
            elif self.ranking:
                self.screen.blit(self.board_image, self.board_rect)
                if self.value is None:
                    self.value = score.Sql.get_score()

                for i in range(len(self.value)):
                    self.screen.blit(self.cups[i], self.cup_rects[i])
                    time_tran = time.strftime(
                            "%Y/%m/%d %H:%M:%S",
                            time.localtime(self.value[i][0])).split()
                    score_text = self.rank_font.render(
                            str(self.value[i][1]), True, (0, 0, 0))
                    time_text1 = self.setting_font.render(
                            time_tran[0], True, (0, 0, 0))
                    time_text2 = self.setting_font.render(
                            time_tran[1], True, (0, 0, 0))
                    self.screen.blit(
                            score_text,
                            (self.cup_rects[i][0]+50, self.cup_rects[i][1]+10)
                            )
                    self.screen.blit(
                            time_text1,
                            (self.cup_rects[i][0]+95, self.cup_rects[i][1]+5)
                            )
                    self.screen.blit(
                            time_text2,
                            (self.cup_rects[i][0]+105, self.cup_rects[i][1]+23)
                            )

                self.screen.blit(self.back_image, self.back_rect)

# ============================================================================
# =================================分享画面====================================
            elif self.share:
                self.screen.blit(self.board_image, self.board_rect)
                self.screen.blit(self.copy_image, self.copy_rect)
                self.screen.blit(self.save_image, self.save_rect)
                self.screen.blit(self.email_image, self.email_rect)
                self.screen.blit(self.back_image, self.back_rect)

# ============================================================================
# ============================================================================
# ================================游戏画面=====================================
            else:
                if not self.pressed:
                    # 绘制小鸟
                    self.screen.blit(
                        self.bird.images[self.bird.image_index(self.delay)],
                        self.bird.rect)
                    # 绘制ready
                    self.screen.blit(self.ready_image, self.ready_rect)
                    # 绘制press开始
                    self.screen.blit(self.press_start_image,
                                     self.press_start_rect)
                else:
                    # 移动小鸟
                    if not self.paused:
                        self.bird.move(self.delay)

                    # 绘制pipe
                    for upipe, dpipe in zip(self.upperpipes, self.lowerpipes):
                        self.screen.blit(upipe.image, upipe.rect)
                        self.screen.blit(dpipe.image, dpipe.rect)

                    # 绘制小鸟
                    self.screen.blit(self.bird.image, self.bird.rect)

                    if self.bird.alive:
                        # 绘制分数
                        score.display(self.screen, self.bg_size, self.score)

                        if not self.paused:
                            # 绘制暂停按钮
                            self.screen.blit(self.pause_image,
                                             self.pause_image_rect)

                            # 移动pipe
                            for upipe, dpipe in zip(self.upperpipes,
                                                    self.lowerpipes):
                                upipe.move()
                                dpipe.move()
                        else:
                            # 绘制继续按钮
                            self.screen.blit(self.resume_image,
                                             self.resume_image_rect)

                    # 生成和删除pipe
                    if 0 < self.upperpipes[0].rect.left < 5:
                        new_upipe, new_dpipe = pipe.get_pipe(
                                self.bg_size, self.land.rect.top)
                        self.upperpipes.append(new_upipe)
                        self.lowerpipes.append(new_dpipe)
                        self.pipe_group.add(new_upipe, new_dpipe)
                    if self.upperpipes[0].rect.right < 0:
                        self.pipe_group.remove(self.upperpipes[0],
                                               self.lowerpipes[0])
                        self.upperpipes.pop(0)
                        self.lowerpipes.pop(0)

                    # 得分
                    if self.bird.alive:
                        for upipe in self.upperpipes:
                            if (upipe.rect.centerx
                                    <= self.bird.rect.centerx
                                    < upipe.rect.centerx + 4):
                                self.score += 1
                                self.sound['point_sound'].play()

                # 检测碰撞
                    # 地面碰撞
                    if (self.bird.alive
                            and self.bird.rect.top + self.bird.rect.height
                            >= self.land.rect.top + 20):
                        self.bird.alive = False
                        self.sound['hit_sound'].play()
                        self.sound['die_sound'].play()

                    # pipe碰撞
                    if (self.bird.alive
                            and pygame.sprite.spritecollide(
                                    self.bird,
                                    self.pipe_group,
                                    False, pygame.sprite.collide_mask
                                    )):
                        self.bird.alive = False
                        self.sound['hit_sound'].play()
                        self.sound['die_sound'].play()

# ===============================游戏结束画面==================================
                    if not self.bird.alive:
                        # 绘制gameover字样
                        self.screen.blit(self.gameover_image,
                                         self.gameover_image_rect)

                        # 绘制成绩面板
                        self.screen.blit(self.score_panel,
                                         self.score_panel_rect)
                        score.show_score(self.screen, self.bg_size, self.score)
                        if not self.recorded and self.value is None:
                            self.value = score.Sql.get_score()
                        if self.value:
                            best_score = self.value[0][1]
                            score.show_best(self.screen,
                                            self.bg_size, best_score)

                        # 绘制奖牌
                        if self.score >= 100:
                            self.screen.blit(self.white_medal,
                                             self.medal_rect)
                        elif self.score >= 60:
                            self.screen.blit(self.gold_medal,
                                             self.medal_rect)
                        elif self.score >= 30:
                            self.screen.blit(self.silver_medal,
                                             self.medal_rect)
                        elif self.score >= 10:
                            self.screen.blit(self.brooze_medal,
                                             self.medal_rect)

                        # 绘制重新开始
                        self.screen.blit(self.retry_image, self.retry_rect)
                        self.screen.blit(self.share_image, self.share_rect)
                        self.screen.blit(self.menu_image, self.menu_rect)

                        # 保存分数
                        if not self.recorded:
                            new_record = score.Sql.set_score(self.score)
                            self.value = score.Sql.get_score()
                            self.recorded = True
                        if new_record:
                            self.screen.blit(self.new_image, self.new_rect)

                    self.screen.blit(self.land.image, self.land.rect)

            self.delay = (self.delay + 1) % 30
# ===========================================================================

            pygame.display.update()
            self.clock.tick(30)

# =============================================================================
# =============================================================================

    def intelligence(self, input_action):
        pygame.event.pump()
        reward = 0.1

        if sum(input_action) != 1:
            raise ValueError("Action Error")

        if input_action[1] == 1:
            if self.bird.rect.top > -2 * self.bird.rect.height:
                self.bird.fly()

        if self.bird.rect.top < 0:
            self.bird.rect.top = 0
            if input_action[1] == 1:
                reward = -0.1

        self.bird.move(self.delay)

        # 移动pipe
        for upipe, dpipe in zip(self.upperpipes, self.lowerpipes):
            upipe.move()
            dpipe.move()

        # 生成和删除pipe
        if 0 < self.upperpipes[0].rect.left < 5:
            new_upipe, new_dpipe = pipe.get_pipe(
                    self.bg_size, self.land.rect.top)
            self.upperpipes.append(new_upipe)
            self.lowerpipes.append(new_dpipe)
            self.pipe_group.add(new_upipe, new_dpipe)
        if self.upperpipes[0].rect.right < 0:
            self.pipe_group.remove(self.upperpipes[0], self.lowerpipes[0])
            self.upperpipes.pop(0)
            self.lowerpipes.pop(0)

        # 得分
        if self.bird.alive:
            for upipe in self.upperpipes:
                if (upipe.rect.centerx
                        <= self.bird.rect.centerx
                        < upipe.rect.centerx + 4):
                    self.score += 1
                    reward = 1

        # 地面碰撞
        if (self.bird.alive
                and self.bird.rect.top + self.bird.rect.height
                >= self.land.rect.top + 20):
            self.bird.alive = False
            self.init_vars(ai=True)
            reward = -1

        # pipe碰撞
        if (self.bird.alive
            and pygame.sprite.spritecollide(
                    self.bird,
                    self.pipe_group,
                    False, pygame.sprite.collide_mask
                    )):
            self.bird.alive = False
            self.init_vars(ai=True)
            reward = -1

        self.screen.blit(self.bg_black, (0, 0))

        # 绘制pipe
        for upipe, dpipe in zip(self.upperpipes, self.lowerpipes):
            self.screen.blit(upipe.image, upipe.rect)
            self.screen.blit(dpipe.image, dpipe.rect)

        # 绘制小鸟
        self.screen.blit(self.bird.image, self.bird.rect)

        # 绘制地面
        self.screen.blit(self.land.image, self.land.rect)
        if self.bird.alive:
            self.land.move()

        self.delay = (self.delay + 1) % 30

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        # score.display(self.screen, self.bg_size, self.score)
        pygame.display.update()

        self.clock.tick(30)
        return image_data, reward, self.bird.alive

# =============================================================================
# =============================================================================


if __name__ == "__main__":
    try:
        game = Game()
        game.play()
#        game.init_vars(ai=True)
#        while game.bird.alive:
#            image_data, reward, _ = game.intelligence([1, 0])
#        pygame.quit()
#        sys.exit(0)
    except SystemExit:
        pass
    except Exception:
        traceback.print_exc()
        pygame.quit()
        input()
