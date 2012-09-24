#!/usr/bin/python

import cv
from cv import *
from connected_components import *
from UnionFind import UnionFind
import random

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
  '''
  metadata = getMetadata('zhjdasn-part2.m4v')
  subtitle_color = metadata['subtitle_color']
  best_portion = metadata['best_portion']
  vstart = metadata['vstart']
  vend = metadata['vend']
  curImg = None
  img = LoadImage('testdata/zhjdasn-extractframe250.png')
  extracted_color_img = extractColor(img, subtitle_color)
  nimg = extracted_color_img
  connectedComponentOutsidePermittedRegionBlacken(extracted_color_img, vstart-5, vend+5)
  #connected_components = connectedComponentLabel(extracted_color_img)
  #blacklist = findComponentsSpanningOutsideRange(connected_components, vstart-5, vend+5)
  #blackenBlacklistedComponents(nimg, connected_components, blacklist)
  SaveImage('extractframe250.png', nimg)
  '''
  for idx,img in iterVideo(vidf):
    if idx != frameno:
      continue
    if idx > frameno:
      break
    img = getBottomQuarter(img)
    SaveImage('extractframe' + str(idx)+'-orig.png', img)
    extracted_color_img = extractColor(img, subtitle_color)
    SaveImage('extractframe' + str(idx)+'-color.png', extracted_color_img)
    #extracted_color_img = extractColor(img, subtitle_color)
    horizontal_boundary = 5
    vertical_boundary = 5
    connectedComponentOutsidePermittedRegionBlacken(extracted_color_img, vstart-vertical_boundary, vend+vertical_boundary)
    SaveImage('extractframe' + str(idx)+'-blackened.png', extracted_color_img)
    vertical_extracted_color_img = extractVertical(extracted_color_img, vstart-vertical_boundary, vend-vstart+vertical_boundary)
    harris = getHarris(vertical_extracted_color_img)
    SaveImage('extractframe' + str(idx)+'-harris.png', harris)
    horizontalActivation = getHorizontalActivationWithHarris(vertical_extracted_color_img, harris)
    hstart,hend = getHorizontalStartEnd(horizontalActivation)
    #nimg = extractHorizontal(vertical_extracted_color_img, hstart, hend-hstart)
    nimg = blackenOutsideHorizontalRegion(vertical_extracted_color_img, hstart-horizontal_boundary, hend-hstart+horizontal_boundary)
    SaveImage('extractframe' + str(idx)+'-horizontalblackened.png', nimg)
    equalizedhist = toEqualizedHistGrayscale(nimg)
    SaveImage('extractframe' + str(idx)+'-equalizehist.png', equalizedhist)
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

if __name__ == '__main__':
  main()
