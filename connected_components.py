#!/usr/bin/python

import cv
from cv import *
from extract_subtitle import *
from UnionFind import UnionFind
import random

def encodeNumAsColor(num):
  color=[0,0,0]
  color[0] = (num & 0x0000FF)
  color[1] = (num & 0x00FF00) >> 8
  color[2] = (num & 0xFF0000) >> 16
  return tuple(color)

def decodeNumFromColor(color):
  return (color[2] << 16) + (color[1] << 8) + (color[0])

def connectedComponentLabel(img):
  labels = UnionFind()
  # pass 1
  for surroundingPoints in iterImgSurrounding1(img):
    currentPoint = surroundingPoints[0]
    y,x = currentPoint
    if sum(img[y,x]) <= 100: # black
      continue
    neighbors = surroundingPoints[1:]
    # if all 4 neighbors are black or unlabeled (ie, 0 labeled white neighbors), assign new label
    # if only 1 neighbor is white, assign its label to current point
    # if more than 1 of the neighbors are white, assign one of their labels to the current point, and note equivalence
    white_neighbors = []
    labeled_white_neighbors = []
    for neighbor in neighbors:
      ny,nx = neighbor
      if sum(img[ny,nx]) > 100: # white neighbor
        white_neighbors.append(neighbor)
        if neighbor in labels: # labeled
          labeled_white_neighbors.append(neighbor)
    if len(labeled_white_neighbors) == 0: # assign new label
      z = labels[currentPoint]
    else:
      for neighbor in labeled_white_neighbors:
        labels.union(neighbor, currentPoint)
      '''
      label = labels[labeled_white_neighbors[0]]
      labels.union(label, currentPoint)
      for neighbor in labeled_white_neighbors[1:]:
        labels.union(label, neighbor)
      '''
  # pass 2
  set_num = 1
  set_to_num = {}
  outimg = CreateImage((img.width, img.height), 8, 3)
  for currentPoint in iterImg(img):
    y,x = currentPoint
    if currentPoint in labels:
      curset = labels[currentPoint]
      #print curset
      if curset not in set_to_num:
        set_to_num[curset] = encodeNumAsColor(set_num)
        set_num += 1
        print set_num
      outimg[y,x] = set_to_num[curset]
    else:
      outimg[y,x] = (0,0,0)
  return outimg

def randomColor():
  return tuple([random.randint(0,255) for i in [0,1,2]])

def visualizeConnectedComponents(connectedComponents, blacklisted_labelnums=frozenset()):
  outimg = CreateImage((connectedComponents.width, connectedComponents.height), 8, 3)
  labelnum_to_color = [(0,0,0)]
  for y,x in iterImg(connectedComponents):
    labelnum = decodeNumFromColor([int(v) for v in connectedComponents[y,x]])
    while labelnum >= len(labelnum_to_color):
      labelnum_to_color.append(randomColor())
    if labelnum in blacklisted_labelnums:
      labelnum = 0
    color = labelnum_to_color[labelnum]
    outimg[y,x] = color
  return outimg

def findComponentsSpanningOutsideRange(connectedComponents, vstart, vend):
  blacklist = set()
  for y,x in iterImg(connectedComponents):
    if y >= vstart and y <= vend:
      continue
    labelnum = decodeNumFromColor([int(v) for v in connectedComponents[y,x]])
    blacklist.add(labelnum)
  return blacklist

def blackenBlacklistedComponents(img, connectedComponents, blacklisted_labelnums):
  for y,x in iterImg(connectedComponents):
    labelnum = decodeNumFromColor([int(v) for v in connectedComponents[y,x]])
    if labelnum in blacklisted_labelnums:
      img[y,x] = (0,0,0)

def connectedComponentOutsidePermittedRegionBlacken(img, vstart, vend):
  labels = UnionFind()
  # pass 1
  blacklist = set()
  for surroundingPoints in iterImgSurrounding1(img):
    currentPoint = surroundingPoints[0]
    y,x = currentPoint
    if sum(img[y,x]) <= 100: # black
      continue
    neighbors = surroundingPoints[1:]
    # if all 4 neighbors are black or unlabeled (ie, 0 labeled white neighbors), assign new label
    # if only 1 neighbor is white, assign its label to current point
    # if more than 1 of the neighbors are white, assign one of their labels to the current point, and note equivalence
    white_neighbors = []
    labeled_white_neighbors = []
    for neighbor in neighbors:
      ny,nx = neighbor
      if sum(img[ny,nx]) > 100: # white neighbor
        white_neighbors.append(neighbor)
        if neighbor in labels: # labeled
          labeled_white_neighbors.append(neighbor)
    if len(labeled_white_neighbors) == 0: # assign new label
      z = labels[currentPoint]
      if y < vstart or y > vend:
        blacklist.add(z)
    else:
      for neighbor in labeled_white_neighbors:
        label = labels[neighbor]
        blacklisted = (label in blacklist)
        labels.union(neighbor, currentPoint)
        if blacklisted:
          blacklist.add(labels[currentPoint])
  # pass 2
  for currentPoint in iterImg(img):
    y,x = currentPoint
    if currentPoint in labels:
      curset = labels[currentPoint]
      if curset in blacklist:
        img[y,x] = (0,0,0)

