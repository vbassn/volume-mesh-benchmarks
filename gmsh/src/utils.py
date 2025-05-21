import numpy as np

def signed_area_xy(polygon):
        """
        Compute the signed area of a polygon given as a list of (x,y,z) tuples.
        Positive => CCW, Negative => CW.
        """
        area = 0.0
        n = len(polygon)
        for i in range(n):
            x1, y1, _ = polygon[i]
            x2, y2, _ = polygon[(i+1) % n]
            area += (x1 * y2) - (x2 * y1)
        return area / 2.0

def ensure_ccw(poly):
    """Ensure polygon is counter-clockwise."""
    if signed_area_xy(poly) < 0:
        return poly[::-1]
    return poly

def ensure_cw(poly):
    """Ensure polygon is clockwise."""
    if signed_area_xy(poly) > 0:
       return poly[::-1]
    return poly
