

import SimpleITK as sitk
import mdai
import pandas as pd
import numpy as np
import cv2
import mainFuncs
import pydicom
import os
import multiprocessing as mp
import functools
from functools import partial
import random



labels_dict = {
    "L_dXjbMB": 1,  # prostate
    "L_Ba5bOB": 2,  # peripheral zone
    "L_8WYbMl": 3,  # central zone
    "L_lPwbJB": 4,  # transition zone
    "L_BnEbY8": 5,  # anterior fibromuscular stroma
    "L_B5mQEB": 6,  # urethra
    "L_8Gq0Kl": 7,  # seminal vesicles L
    "L_8EpQ0d": 8,  # seminal vesicles R
    "L_80JOXl": 9,  # lymph node regional
    "L_lA2Mkd": 10,  # lymph node regional group
    "L_837Z58": 11,  # obturator A
    "L_BkK2ZB": 12,  # internal iliac A
    "L_dDZPQB": 13,  # obturator
    "L_lbEN3B": 14,  # internal iliac
    "L_BL7pO8": 15,  # iexternal iliac A
    "L_dXnmOB": 16,  # iexternal iliac B
    "L_8J5ezd": 17,  # lesion 1
    "L_8eE6g8": 18,  # lesion 2
    "L_84r7kl": 19,  # lesion 3
    "L_8KGNn8": 20,  # lesion 4
    "L_Bo2aM8": 21,  # lesion 5
    "L_89vrVB": 22,  # curvilinear contact
    # Label Group, 'G_dW4mY5': JD
    "L_BOaoRd": 100,  # prostate
    "L_dp6yvB": 101,  # peripheral zone
    "L_lVg7Kd": 102,  # central zone
    "L_837KW8": 103,  # transition zone
    "L_BkKnDB": 104,  # anterior fibromuscular stroma
    "L_dDZw2B": 105,  # urethra
    "L_lbE6eB": 106,  # seminal vesicles L
    "L_BL7bn8": 107,  # seminal vesicles R
    "L_Bw52Zd": 108,  # lymph node regional
    "L_dXnyqB": 109,  # lymph node regional group
    "L_BaZzwl": 110,  # obturator A
    "L_8WArNl": 111,  # internal iliac A
    "L_lPMoOd": 112,  # obturator
    "L_Bn9ro8": 113,  # internal iliac
    "L_B5YpOB": 114,  # iexternal iliac A
    "L_8G9awl": 115,  # iexternal iliac B
    "L_B7jzZ8": 116,  # lesion 1
    "L_Br9NG8": 117,  # lesion 2
    "L_By7q7d": 118,  # lesion 3
    "L_djon38": 119,  # lesion 4
    "L_dMrVV8": 120,  # lesion 5
    "L_89vxVB": 121,  # curvilinear contact
    # Label Group, 'G_2PXqa2': TL
    "L_lzr1Ql": 200,  # prostate
    "L_l152el": 201,  # peripheral zone
    "L_dQeOkd": 202,  # central zone
    "L_Bn9rv8": 203,  # transition zone
    "L_B5YpAB": 204,  # anterior fibromuscular stroma
    "L_8G9aOl": 205,  # urethra
    "L_B7jz68": 206,  # seminal vesicles L
    "L_Br9NY8": 207,  # seminal vesicles R
    "L_By7qQd": 208,  # lymph node regional
    "L_djonL8": 209,  # lymph node regional group
    "L_dMrVE8": 210,  # obturator A
    "L_89vx7B": 211,  # internal iliac A
    "L_lzr1Dl": 212,  # obturator
    "L_l152Gl": 213,  # internal iliac
    "L_dQeO3d": 214,  # iexternal iliac A
    "L_BNwY0l": 215,  # iexternal iliac B
    "L_lRAVoB": 216,  # lesion 1
    "L_lqjXgd": 217,  # lesion 2
    "L_8ZGAKB": 218,  # lesion 3
    "L_B6j2DB": 219,  # lesion 4
    "L_BmWPxd": 220,  # lesion 5
    "L_dvnxRd": 221,  # curvilinear contact
    # Label Group, 'G_d9Pj0K': AZ
    "L_8J5R9d": 300,  # prostate
    "L_8eEaM8": 301,  # peripheral zone
    "L_84r04l": 302,  # central zone
    "L_8KGk98": 303,  # transition zone
    "L_Bo2yW8": 304,  # anterior fibromuscular stroma
    "L_8Y5ab8": 305,  # urethra
    "L_8xGRyd": 306,  # seminal vesicles L
    "L_d2GgN8": 307,  # seminal vesicles R
    "L_BgvaL8": 308,  # lymph node regional
    "L_8E9Vk8": 309,  # lymph node regional group
    "L_80n2Ol": 310,  # obturator A
    "L_lAGVp8": 311,  # internal iliac A
    "L_BOag0d": 312,  # obturator
    "L_dp601B": 313,  # internal iliac
    "L_lVgRzd": 314,  # iexternal iliac A
    "L_837oa8": 315,  # iexternal iliac B
    "L_BkKxaB": 316,  # lesion 1
    "L_dDZvyB": 317,  # lesion 2
    "L_lbEjEB": 318,  # lesion 3
    "L_BL7xb8": 319,  # lesion 4
    "L_Bw57pd": 320,  # lesion 5
    "L_dXnaaB": 321,  # curvilinear contact
    # Label Group, 'G_2BeMO5': TK
    "L_BaZy2l": 400,  # prostate
    "L_8WAVql": 401,  # peripheral zone
    "L_lPMmkd": 402,  # central zone
    "L_Bn9qv8": 403,  # transition zone
    "L_B5Y6AB": 404,  # anterior fibromuscular stroma
    "L_8G9pOl": 405,  # urethra
    "L_B7jg68": 406,  # seminal vesicles L
    "L_Br9aY8": 407,  # seminal vesicles R
    "L_By7yQd": 408,  # lymph node regional
    "L_djo1L8": 409,  # lymph node regional group
    "L_dMrmE8": 410,  # obturator A
    "L_89vV7B": 411,  # internal iliac A
    "L_lzrjDl": 412,  # obturator
    "L_l150Gl": 413,  # internal iliac
    "L_dQeP3d": 414,  # iexternal iliac A
    "L_BNwm0l": 415,  # iexternal iliac B
    "L_lRAwoB": 416,  # lesion 1
    "L_lqjagd": 417,  # lesion 2
    "L_8ZGRKB": 418,  # lesion 3
    "L_B6jLDB": 419,  # lesion 4
    "L_BmW2xd": 420,  # lesion 5
    "L_dvnGRd": 421,  # curvilinear contact
    # Label Group, 'G_Kp0aRK': LK
    "L_B5YEEB": 500,  # prostate
    "L_8G9xKl": 501,  # peripheral zone
    "L_B7jEp8": 502,  # central zone
    "L_Br9r48": 503,  # transition zone
    "L_By7QDd": 504,  # anterior fibromuscular stroma
    "L_djo4Z8": 505,  # urethra
    "L_dMrLb8": 506,  # seminal vesicles L
    "L_89vP0B": 507,  # seminal vesicles R
    "L_lzrNRl": 508,  # lymph node regional
    "L_l15Ekl": 509,  # lymph node regional group
    "L_dQeLRd": 510,  # obturator A
    "L_BNwL9l": 511,  # internal iliac A
    "L_lRALgB": 512,  # obturator
    "L_lqj79d": 513,  # internal iliac
    "L_8ZGLYB": 514,  # iexternal iliac A
    "L_B6j0YB": 515,  # iexternal iliac B
    "L_BmW4Nd": 516,  # lesion 1
    "L_dvn9md": 517,  # lesion 2
    "L_8J5xOd": 518,  # lesion 3
    "L_8eE4b8": 519,  # lesion 4
    "L_84rEQl": 520,  # lesion 5
    "L_8KGxb8": 521,  # curvilinear contact
}

"""
downloading data from mdai client
"""
mdai_client = mdai.Client(domain='public.md.ai', access_token="1d48dd3c05ce6d59759915d9328fe769")
p = mdai_client.project('gaq3y0Rl', path='/workspaces/konwersjaJsonData/out')
p.set_labels_dict(labels_dict)
datasetId='D_gQm1nQ'
dataset = p.get_dataset_by_id(datasetId)
dataset.prepare()