#!/usr/bin/python

import cv
from cv import *
from extract_subtitle import *
from UnionFind import UnionFind
import random
from connected_components import *

def main():
  vidf = 'video.m4v'
  if len(sys.argv) > 1:
    vidf = sys.argv[1]
  frameno = 0
  if len(sys.argv) > 2:
    frameno = int(sys.argv[2])
  metadata = getMetadata(vidf)
  subtitle_color = metadata['subtitle_color']
  best_portion = metadata['best_portion']
  vstart = metadata['vstart']
  vend = metadata['vend']
  curImg = None
  for idx,img in iterVideo(vidf):
    if idx != frameno:
      continue
    if idx > frameno:
      break
    img = getBottomQuarter(img)
    #extracted_color_img = extractColor(img, subtitle_color)
    extracted_color_img = extractColor(img, subtitle_color)
    connectedComponentOutsidePermittedRegionBlacken(extracted_color_img, vstart-5, vend+5)
    #connected_components = connectedComponentLabel(extracted_color_img)
    #blacklist = findComponentsSpanningOutsideRange(connected_components, vstart-5, vend+5)
    #nimg = visualizeConnectedComponents(connected_components, blacklist)
    '''
    vertical_extracted_color_img = extractVertical(extracted_color_img, vstart, vend-vstart)
    harris = getHarris(vertical_extracted_color_img)
    horizontalActivation = getHorizontalActivationWithHarris(vertical_extracted_color_img, harris)
    hstart,hend = getHorizontalStartEnd(horizontalActivation)
    #nimg = extractHorizontal(vertical_extracted_color_img, hstart, hend-hstart)
    nimg = blackenOutsideHorizontalRegion(vertical_extracted_color_img, hstart, hend-hstart)
    '''
    nimg = extracted_color_img
    SaveImage('extractframe' + str(idx)+'.png', nimg)

if __name__ == '__main__':
  main()
