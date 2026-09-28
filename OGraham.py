# Copyright (c) 2026 Phan Thanh An and Tran Anh Dung
# Institute of Mathematical and Computational Sciences (IMACS)
# Contact: Prof. Phan Thanh An <thanhan@hcmut.edu.vn>
#          Tran Anh Dung <trananhdung@iuh.edu.vn>, <tadung.sdh231@hcmut.edu.vn>
# Released under the MIT License (see LICENSE).
# OGraham.py
# Connected orthogonal convex hull (COCH) of a finite planar point set by the
# modified Graham scan of P.T. An (Phan Thanh An), P.T.T. Huyen and N.T. Le,
# Applied Mathematics and Computation 397 (2021) 125889
# (doi 10.1016/j.amc.2020.125889). Changes to the O-Graham code are marked [COCH-fix]:
#   - the figure that was opened at every call (fig, ax = plt.subplots()) is removed;
#   - the scan of each of the four sets now includes its last point (it was
#     range(len(set)-1)); without it the points qq1, qq2, qq3, qq4 at the ends of the
#     extreme edges were skipped and the hull could leave out part of the input set.
import numpy as np

#==================================BEGIN=====================================#

# Read input points

#points = np.genfromtxt('/Users/user/Documents/Code/Crack/Code_python/code/Images/10.jpg', delimiter=',')



# Time measurement starts here



#=============== Start preprocessing step ==================#

#Find 8 special points
def o_graham(points):
    points = np.array([np.array(l) for l in points])
    
    maxY = np.nanmax(points[:,1])
    minY = np.nanmin(points[:,1])
    maxX = np.nanmax(points[:,0])
    minX = np.nanmin(points[:,0])


    rightPoints = points[np.where(np.equal(points[:,0], maxX))]
    leftPoints = points[np.where(np.equal(points[:,0], minX))]
    topPoints = points[np.where(np.equal(points[:,1], maxY))]
    bottomPoints = points[np.where(np.equal(points[:,1], minY))]

    if len(topPoints) == 1:
        top = (topPoints[0],)
    else:
        topPoints = sorted(topPoints, key = lambda x : x[0])
        top = (topPoints[0], topPoints[len(topPoints)-1])

        # bottom
    if len(bottomPoints) == 1:
        bottom = (bottomPoints[0],)
    else:
        bottomPoints = sorted(bottomPoints, key = lambda x : -x[0])
        bottom = (bottomPoints[0], bottomPoints[len(bottomPoints)-1])

        # right
    if len(rightPoints) == 1:
       right = (rightPoints[0],)
    else:
        rightPoints = sorted(rightPoints, key = lambda x : -x[1])
        right = (rightPoints[0], rightPoints[len(rightPoints)-1])

        # left
    if len(leftPoints) == 1:
        left = (leftPoints[0],)
    else:
        leftPoints = sorted(leftPoints, key = lambda x : x[1])
        left = (leftPoints[0], leftPoints[len(leftPoints)-1])

    if len(top) == 1:
        q1 = qq4 = top[0]
    else:
        q1 = top[0]
        qq4 = top[1]
    q4 = right[0]
    if len(right) == 1:
        qq3 = right[0]
    else:
        qq3 = right[1]
    q3 = bottom[0]
    if len(bottom) == 1:
        qq2 = bottom[0]
    else:
        qq2 = bottom[1]
    q2 = left[0]
    if len(left) == 1:
        qq1 = left[0]
    else:
        qq1 = left[1]

    # Separate to 4 sets of points

    set1 = points[np.where((points[:,0] <= q1[0]) &  (points[:,1] >= qq1[1]))]
    set2 = points[np.where((points[:,0] <= qq2[0]) &  (points[:,1]  <= q2[1]))]
    set3 = points[np.where((points[:,0] >= q3[0]) &  (points[:,1]  <= qq3[1]))]
    set4 = points[np.where((points[:,0] >= qq4[0]) &  (points[:,1] >= q4[1]))]

    #=============== Finish the preprocessing step ==================#



    #===== Start to find o.convex by O-Graham algorithm ===========#


    ############################################################
    set1 = np.array( sorted( set1, key = lambda x: (-x[1], x[0]) ) )

    b = [True]
    for i in range( len(set1) - 1 ):
        if set1[i][1] == set1[i+1][1]:
            b.append( False )
        else:
            b.append( True )

    set1 = set1[ b ]

    ############################################################
    set2 = np.array( sorted( set2, key = lambda x: (x[0], x[1]) ) )


    b = [True]
    for i in range( len(set2) - 1 ):
        if set2[i][0] == set2[i+1][0]:
            b.append( False )
        else:
            b.append( True )

    set2 = set2[ b ]


    ############################################################
    set3 = np.array( sorted( set3, key = lambda x: (x[1], -x[0]) ) )


    b = [True]
    for i in range( len(set3) - 1 ):
        if set3[i][0] == set3[i+1][0]:
            b.append( False )
        else:
            b.append( True )

    set3 = set3[ b ]


    ############################################################
    set3 = np.array( sorted( set3, key = lambda x: (x[1], -x[0]) ) )


    b = [True]
    for i in range( len(set3) - 1 ):
        if set3[i][0] == set3[i+1][0]:
            b.append( False )
        else:
            b.append( True )


    set3 = set3[ b ]


    ############################################################
    set4 = np.array( sorted( set4, key = lambda x: (-x[0], -x[1]) ) )


    b = [True]
    for i in range( len(set4) - 1 ):
        if set4[i][0] == set4[i+1][0]:
            b.append( False )
        else:
            b.append( True )

    set4 = set4[ b ]

    ############################################################

    hull = [set1[0]]
    for i in range(len(set1)):   # [COCH-fix] was range(len(set1)-1): the last point of each set (qq1, qq2, qq3, qq4) was skipped
        cand = hull[-1]
        if set1[i][0] < cand[0]:
            hull.append(set1[i])

    for i in range(len(set2)):   # [COCH-fix] was range(len(set2)-1): the last point of each set (qq1, qq2, qq3, qq4) was skipped
        cand = hull[-1]
        if set2[i][1] < cand[1]:
            hull.append(set2[i])

    for i in range(len(set3)):   # [COCH-fix] was range(len(set3)-1): the last point of each set (qq1, qq2, qq3, qq4) was skipped
        cand = hull[-1]
        if set3[i][0] > cand[0]:
            hull.append(set3[i])


    for i in range(len(set4)):   # [COCH-fix] was range(len(set4)-1): the last point of each set (qq1, qq2, qq3, qq4) was skipped
        cand = hull[-1]
        if set4[i][1] > cand[1]:
            hull.append(set4[i])
    hull.append(qq4)
    hull.append(q1)

    #============== plot the c.o. convex hull ===============#

    # [COCH-fix] fig, ax = plt.subplots() removed: it opened a new figure at every call
    if hull is not None:
            # plot the convex hull boundary, extra iteration at
            # the end so that the bounding line wraps around
        #hull.append(hull[0])
        S = []
        n = len(hull)
            #print("number points to plot: ", n)
        for i in range(0, n-1):
            if hull[i+1][0] > hull[i][0] and hull[i+1][1] > hull[i][1]:
                p3 = [hull[i][0], hull[i+1][1]]
                S.append([i+1, p3])
            elif hull[i+1][0] > hull[i][0] and hull[i+1][1] < hull[i][1]:
                p3 = [hull[i+1][0], hull[i][1]]
                S.append([i+1, p3])
            elif hull[i+1][0] < hull[i][0] and hull[i+1][1] < hull[i][1]:
                p3 = [hull[i][0], hull[i+1][1]]
                S.append([i+1, p3])
            elif hull[i+1][0] < hull[i][0] and hull[i+1][1] > hull[i][1]:
                p3 = [hull[i+1][0], hull[i][1]]
                S.append([i+1, p3])




        for i in range(len(S)):
            hull.insert(S[i][0]+i, S[i][1])

    hull.append(np.array(hull[0]))
    
    hull = np.array(hull)[::-1]
    
    return hull


