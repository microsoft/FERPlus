#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

import math
  
class Point(object):
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y
     
    def __add__(self, p):
        """Point(x1+x2, y1+y2)"""
        return Point(self.x+p.x, self.y+p.y)
     
    def __sub__(self, p):
        """Point(x1-x2, y1-y2)"""
        return Point(self.x-p.x, self.y-p.y)
     
    def __mul__( self, scalar ):
        """Point(x1*x2, y1*y2)"""
        return Point(self.x*scalar, self.y*scalar)
     
    def __div__(self, scalar):
        """Point(x1/x2, y1/y2)"""
        return Point(self.x/scalar, self.y/scalar)
     
    def __str__(self):
        return "(%s, %s)" % (self.x, self.y)
     
    def length(self):
        return math.sqrt(self.x**2 + self.y**2)
     
    def distance_to(self, p):
        """Calculate the distance between two points."""
        return (self - p).length()
     
    def as_tuple(self):
        """(x, y)"""
        return (self.x, self.y)
     
    def clone(self):
        """Return a full copy of this point."""
        return Point(self.x, self.y)
     
    def integerize(self):
        """Convert co-ordinate values to integers."""
        self.x = int(self.x+0.5)
        self.y = int(self.y+0.5)
     
    def floatize(self):
        """Convert co-ordinate values to floats."""
        self.x = float(self.x)
        self.y = float(self.y)
     
    def reset(self, x, y):
        """Reset x & y coordinates."""
        self.x = x
        self.y = y
     
    def shift(self, pt):
        """Move to new (x+pt.x,y+pt.y)."""
        self.x = self.x + pt.x
        self.y = self.y + pt.y
     
    def shift_xy(self, dx, dy):
        """Move to new (x+dx,y+dy)."""
        self.x = self.x + dx
        self.y = self.y + dy
     
    def rotate(self, rad):
        """Rotate counter-clockwise by rad radians.
        Positive y goes *up,* as in traditional mathematics.
        The new position is returned as a new Point.
        """
        s, c = [f(rad) for f in (math.sin, math.cos)]
        x, y = (c*self.x - s*self.y, s*self.x + c*self.y)
        return Point(x,y)
     
    def rotate_about(self, p, theta):
        """Rotate counter-clockwise around a point, by theta degrees.
        Positive y goes *up,* as in traditional mathematics.
        The new position is returned as a new Point.
        """
        result = self.clone()
        result.shift(-p.x, -p.y)
        result.rotate(theta)
        result.shift(p.x, p.y)
        return result
  
class Rect(object):
    """The rectangle stores left, top, right, and bottom values.
    Coordinates are based on screen coordinates.
    origin                            top
    +-----> x increases                |
    |                           left  -+-  right
    v                                  |
    y increases                      bottom
    """

    def __init__(self, box):
        """Initialize a rectangle from two points."""
        self.left = box[0]
        self.top = box[1]
        self.right = box[2]
        self.bottom = box[3]
 
    def as_tuple(self):
        """(left, top, right, bottom)"""
        return (self.left, self.top, self.right, self.bottom)

    def width(self): 
        """Width"""
        return (self.right - self.left) 

    def height(self): 
        """Height"""
        return (self.bottom - self.top) 

    def contains(self, pt):
        """Return true if a point is inside the rectangle."""
        x,y = pt.as_tuple()
        return (self.left <= x <= self.right and
                self.top <= y <= self.bottom)

    def shift(self, pt):
        """Shift by pt.x and pt.y."""
        self.left = self.left + pt.x
        self.right = self.right + pt.x
        self.top = self.top + pt.y
        self.bottom = self.bottom + pt.y

    def shift_xy(self, dx, dy): 
        """Shift by dx and dy."""
        self.left = self.left + dx
        self.right = self.right + dx
        self.top = self.top + dy
        self.bottom = self.bottom + dy
 
    def equal(self, other): 
        """Return true if a rectangle is identical to this rectangle."""
        return (self.right == other.left and self.left == other.right and
                self.top == other.bottom and self.bottom == other.top)
 
    def overlaps(self, other):
        """Return true if a rectangle overlaps this rectangle."""
        return (self.right > other.left and self.left < other.right and
                self.top < other.bottom and self.bottom > other.top)

    def intersect(self, other): 
        """Return the intersect rectangle.
        Note we don't check here whether the intersection is valid
        If needed, call overlaps() first to check 
        """ 
        return Rect((max(self.left, other.left), 
                    max(self.top, other.top), 
                    min(self.right, other.right), 
                    min(self.bottom, other.bottom)))
    
    def clamp(self, xmin, ymin, xmax, ymax): 
        """Return clamped rectangle based on the other rectangle.
        Note we don't check here whether the output is valid
        If needed, call overlaps() first to check 
        """
        self.left = max(self.left, xmin)
        self.right = min(self.right, xmax)
        self.top = max(self.top, ymin)
        self.bottom = min(self.bottom, ymax)         

    def top_left(self):
        """Return the top-left corner as a Point."""
        return Point(self.left, self.top)
     
    def bottom_right(self):
        """Return the bottom-right corner as a Point."""
        return Point(self.right, self.bottom)

    def center(self): 
        """Return the center as a Point.""" 
        return Point((self.left+self.right)/2.0, (self.top+self.bottom)/2.0) 

    def mult(self, xmul, ymul): 
        """Return a rectangle with all coordinates multipled by a number.""" 
        return Rect((self.left*xmul, self.top*ymul, self.right*xmul, self.bottom*ymul))
     
    def scale(self, scale):
        """Return a scaled rectangle with identical center."""
        xctr = (self.left + self.right)/2.0
        yctr = (self.top + self.bottom)/2.0
        width = self.width()*scale
        height = self.height()*scale
        xstart = xctr-width/2.0
        ystart = yctr-height/2.0
        return Rect((xstart, ystart, xstart+width, ystart+height))

    def cocenter(self, new_width, new_height): 
        """Return a new rectangle with identical center."""
        xctr = (self.left + self.right)/2.0
        yctr = (self.top + self.bottom)/2.0
        xstart = xctr - new_width/2.0
        ystart = yctr - new_height/2.0
        return Rect((xstart, ystart, xstart+new_width, ystart+new_height))

    def integerize(self):
        """Convert co-ordinate values to integers."""
        self.left = int(self.left+0.5)
        self.right = int(self.right+0.5)
        self.top = int(self.top+0.5)
        self.bottom = int(self.bottom+0.5)
     
    def floatize(self):
        """Convert co-ordinate values to floats."""
        self.left = float(self.left)
        self.right = float(self.right)
        self.top = float(self.top)
        self.bottom = float(self.bottom)
     
    def __str__( self ):
        return "<Rect (%s,%s)-(%s,%s)>" % (self.left,self.top,
                                        self.right,self.bottom)
     