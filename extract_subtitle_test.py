#!/usr/bin/python

from extract_subtitle import *

import unittest

class TestExtractSubtitle(unittest.TestCase):
  def test_getVerticalStartEnd(self):
    activation = [0, 0, 0, 1, 1, 1, 0, 0, 0]
    start,end = getVerticalStartEnd(activation)
    self.assertEqual(3, start)
    self.assertEqual(5, end)

if __name__ == '__main__':
  unittest.main()

