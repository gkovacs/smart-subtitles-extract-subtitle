#!/usr/bin/python

from extract_subtitle import *

import unittest

class TestExtractSubtitle(unittest.TestCase):
  def test_getVerticalStartEnd(self):
    activation = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    start,end = getVerticalStartEnd(activation)
    self.assertEqual(3, start)
    self.assertEqual(5, end)

  def test_haveTransition(self):
    img1a = LoadImage('testdata/hzgg-1a.png')
    img1b = LoadImage('testdata/hzgg-1b.png')
    img2a = LoadImage('testdata/hzgg-2a.png')
    img2b = LoadImage('testdata/hzgg-2b.png')
    img3a = LoadImage('testdata/hzgg-3a.png')
    img3b = LoadImage('testdata/hzgg-3b.png')
    img4a = LoadImage('testdata/zhjdasn-4a.png')
    img4b = LoadImage('testdata/zhjdasn-4b.png')
    self.assertFalse(haveTransition(img1a, img1b))
    self.assertFalse(haveTransition(img2a, img2b))
    self.assertFalse(haveTransition(img3a, img3b))
    self.assertFalse(haveTransition(img4a, img4b))
    self.assertTrue(haveTransition(img1a, img2a))
    self.assertTrue(haveTransition(img2a, img3a))

if __name__ == '__main__':
  unittest.main()

