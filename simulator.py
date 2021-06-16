#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import ray
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


class GestureKeyboardSimulator:
    def __init__(self):
        self.keyboard = [
            ["q", "w", "e", "r", "t", "y", "u", "i", "o",  "p"],
            ["a", "s", "d", "f", "g", "h", "j", "k", "l"],
            ["z", "x", "c", "v", "b", "n", "m"]
        ]
        self.key_height = 1
        self.key_width = 1
        self.row_offset = 0.5
        self.keyboard_height = len(self.keyboard) * self.key_height
        self.keyboard_width = len(self.keyboard[0]) * self.key_width

    def get_key_index(self, key):
        for i, row in enumerate(self.keyboard):
            if key in row:
                for j, col in enumerate(row):
                    if key == col:
                        return (i, j)

    def get_key_center_position(self, key):
        row, col = self.get_key_index(key)
        x_position = row * self.row_offset + (col + 0.5) * self.key_width
        y_position = (row + 0.5) * self.key_height
        return (x_position, y_position)

    def get_gesture_path(self, word):
        gesture_path = []
        for i in range(len(word)):
            key = word[i]
            key_x_center, key_y_center = self.get_key_center_position(key)
            gesture_path.append((key_x_center, key_y_center))

        return gesture_path


    def draw_gesture_path(self, gesture_path):
        fig, ax = plt.subplots()

        for i in range(len(gesture_path) - 1):
            start_key_x_center, start_key_y_center = gesture_path[i]
            end_key_x_center, end_key_y_center = gesture_path[i + 1]
            ax.plot(
                [start_key_x_center, end_key_x_center],
                [self.keyboard_height - start_key_y_center, self.keyboard_height - end_key_y_center],
            )


        for i, row in enumerate(self.keyboard):
            for j, col in enumerate(row):
                left = i * self.row_offset + j * self.key_width
                bottom = (self.keyboard_height - 1)- i * self.key_height
                ax.add_patch(
                    patches.Rectangle(
                        (left, bottom),
                        self.key_width,
                        self.key_height,
                        linewidth=1,
                        edgecolor='black',
                        facecolor='none'
                    )
                )
                key_x_center, key_y_center = self.get_key_center_position(col)
                ax.text(
                    key_x_center,
                    self.keyboard_height - key_y_center,
                    col,
                    horizontalalignment='center',
                    verticalalignment='center'
                )

        plt.show()

    def init(self, word_list_path):
        self.word_list = {}
        with open(word_list_path) as f_r:
            lines = f_r.read().splitlines()
            for line in lines:
                word = line.split("\t")[0]
                frequency = float(line.split("\t")[1])
                gesture_path = gks.get_gesture_path(word)
                self.word_list[word] = {
                    "frequency": frequency,
                    "template": gesture_path
                }

    def draw_template(self, word):
        self.draw_gesture_path(self.word_list[word]["template"])

    @ray.remote
    def dtw(source, target, word, freq):
        dist, path = fastdtw(
            np.asarray(source),
            np.asarray(target),
            dist=euclidean
        )
        return (word, dist, freq)

    def get_word_suggestion(self, path, word_list, suggestion_num):
        results = ray.get([self.dtw.remote(
            path, value["template"], key, value["frequency"]
        ) for key, value in word_list])
        score_list = self.get_score(results)
        closest_words = list(
            map(
                lambda item: item[0],
                sorted(score_list.items(), key=lambda item: item[1], reverse=True)
            )
        )
        return closest_words[:suggestion_num]

    def get_closest_keys(self, pos, dist_thres):
        dist_list = {}
        for i, row in enumerate(self.keyboard):
            for j, col in enumerate(row):
                x_pos = i * self.row_offset + (j + 0.5) * self.key_width
                y_pos = (i + 0.5) * self.key_height
                dist = math.sqrt(
                    (pos[0] - x_pos) ** 2 + (pos[1] - y_pos) ** 2
                )
                dist_list[col] = dist
        filtered_dist_list = list(
            filter(
                lambda item: True if item[1] <= dist_thres else False,
                dist_list.items()
            )
        )
        closest_keys = list(
            map(
                lambda item: item[0],
                sorted(filtered_dist_list, key=lambda item: item[1])
            )
        )
        return closest_keys

    def get_filtered_word_list(self, path, mode, word):
        first_pos = self.get_key_center_position(word[0])
        last_pos = self.get_key_center_position(word[-1])
    
        first_pos_closest_keys = self.get_closest_keys(first_pos, 1)
        last_pos_closest_keys = self.get_closest_keys(last_pos, 1)

        if mode == 0: # first + last
            return list(
                filter(
                    lambda item: True if (item[0][0] == word[0]) and (item[0][-1] == word[-1]) else False,
                    self.word_list.items()
                )
            )
        elif mode == 1: # first
            return list(
                filter(
                    lambda item: True if (item[0][0] == word[0]) and (item[0][-1] in last_pos_closest_keys) else False,
                    self.word_list.items()
                )
            )
        elif mode == 2: # last
            return list(
                filter(
                    lambda item: True if (item[0][0] in first_pos_closest_keys) and (item[0][-1] == word[-1]) else False,
                    self.word_list.items()
                )
            )
        elif mode == 3: # none
            return list(
                filter(
                    lambda item: True if (item[0][0] in first_pos_closest_keys) and (item[0][-1] in last_pos_closest_keys) else False,
                    self.word_list.items()
                )
            )

    def add_noise(self, path, standard_error, noise_num):
        noise_path = []
        standard_deviation = standard_error * math.sqrt(noise_num)
        for point in path: 
            noise_x = np.random.normal(point[0], standard_deviation, noise_num)
            noise_y = np.random.normal(point[1], standard_deviation, noise_num)
            noise_points = [point]
            for i in range(noise_num):
                noise_points.append((
                    noise_x[i],
                    noise_y[i]
                ))
            random.shuffle(noise_points)
            noise_path = noise_path + noise_points
        return noise_path

    def print_result(self, result):
        for key, value in result.items():
            is_in = [1 if item != -1 else 0 for item in value]
            avg_rank = list(
                filter(
                    lambda item: True if (item != -1) else False,
                    value
                )
            )
            print(key, round(np.average(is_in), 2), round(np.average(avg_rank), 2))

    def get_score(self, results):
        alpha = 0.95
        sum_r = 0.0
        sum_n = 0.0
        score_list = {}
        for word, dist, freq in results:
            r = 1.0/(1.0+dist)
            sum_r = sum_r+r
            n = freq
            sum_n = sum_n+n
        for word, dist, freq in results:
            r = 1.0/(1.0+dist)
            n = freq
            score_list[word] = (alpha*r/sum_r)+((1-alpha)*n/sum_n)
        return score_list

    def simulation(self):
        file_name = "simulation_result.txt"
        simulated_words = []
        if os.path.exists(file_name):
            with open(file_name, "r") as f:
                lines = f.read().splitlines()
                print(len(lines))
                for line in lines:
                    word = line.split("\t")[0]
                    simulated_words.append(word)
        else:
            print(len(simulated_words))

        simulation_result = {}
        for mode in range(4):
            simulation_result[mode] = []

        noise_se = 0.2
        noise_num_per_point = 10

        for key, value in self.word_list.items():
            if key in simulated_words:
                continue
            path = self.add_noise(value["template"], noise_se, noise_num_per_point)
            match_index_list = [key]
            for mode in range(4):
                filtered_word_list = self.get_filtered_word_list(path, mode, key)
                suggested_words = self.get_word_suggestion(
                    path,
                    filtered_word_list,
                    10
                )
                try:
                    match_index = suggested_words.index(key)
                except:
                    match_index = -1
                simulation_result[mode].append(match_index)
                match_index_list.append(str(match_index))
            with open(file_name, "a") as f:
                f.write("\t".join(match_index_list) + "\n")
            simulated_words.append(key)

            if len(simulation_result[mode]) % 10 == 0:
                print(len(simulation_result[mode]), len(self.word_list.items()))
                self.print_result(simulation_result)

        print("final")
        self.print_result(simulation_result)

ray.init()
gks = GestureKeyboardSimulator()
gks.init("word_list.txt")
gks.simulation()
