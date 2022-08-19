import numpy as np
from numpy import sin, cos, arctan2
from itertools import cycle
from sys import argv, exit
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui



class InvertedPendulum(QtGui.QWidget):
    '''Constans init:
    M - weight of the cart
    m - weight of the ball
    l - the length of the pendulum arm

    Initial conditions:
    x0 - the initial position of cart
    dx0 - initial cart speed
    theta0 - initial position of pendulum
    dtheta0 - initial speed of pendulum

    External disturbances:
    dis_cyc - this variable is responsible whether the disturbance is looped
    disruption - disturbance values at succesive time moments

    Board parametres:
    iw, ih - width and height
    x_max - max horizontal coordinate (oś x jest symetryczna, więc minimalna wynosi -x_max)
    h_min - min vertical coordinate
    h_max - max vertical coordinate

    The above data is taken from the file if f_name is not empty'''

    def __init__(self, M=10, m=5, l=50, x0=0, theta0=0, dx0=0, dtheta0=0, dis_cyc=True, disruption=[0], iw=1000, ih=500, x_max=100, h_min=0, h_max=100, f_name="3.txt"):

        if f_name:
            with open(f_name) as f_handle:
                lines = f_handle.readlines()
                init_cond = lines[0].split(' ')
                self.M, self.m, self.l, self.x0, self.theta0, self.dx0, self.dtheta0 = [float(el) for el in init_cond[:7]]
                self.image_w, self.image_h, self.x_max, self.h_min, self.h_max = [int(el) for el in init_cond[-5:]]
                if lines[1]:
                    self.disruption = cycle([float(el) for el in lines[2].split(' ')])
                else:
                    self.disruption = iter([float(el) for el in lines[2].split(' ')])
        else:
            self.M, self.m, self.l, self.x0, self.theta0, self.dx0, self.dtheta0 = M, m, l, x0, theta0, dx0, dtheta0
            self.image_w, self.image_h, self.x_max, self.h_min, self.h_max = iw, ih, x_max, h_min, h_max
            if dis_cyc:
                self.disruption = cycle(disruption)
            else:
                self.disruption = iter(disruption)
        super(InvertedPendulum, self).__init__(parent=None)


    # Board init
    def init_image(self):
        self.h_scale = self.image_h/(self.h_max-self.h_min)
        self.x_scale = self.image_w/(2*self.x_max)
        self.hor = (self.h_max-10)*self.h_scale
        self.c_w = 16*self.x_scale
        self.c_h = 8*self.h_scale
        self.r = 8
        self.x = self.x0
        self.theta = self.theta0
        self.dx = self.dx0
        self.dtheta = self.dtheta0
        self.setFixedSize(self.image_w, self.image_h)
        self.show()
        self.setWindowTitle("Inverted Pendulum")
        self.update()

    # Drawing of pendulum and scale
    def paintEvent(self, e):
        x, x_max, x_scale, theta = self.x, self.x_max, self.x_scale, self.theta
        hor, l, h_scale = self.hor, self.l, self.h_scale
        image_w, c_w, c_h, r, image_h, h_max, h_min = self.image_w, self.c_w, self.c_h, self.r, self.image_h, self.h_max, self.h_min
        painter = QtGui.QPainter(self)
        painter.setPen(pg.mkPen('k', width=2.0*self.h_scale))
        painter.drawLine(0, hor, image_w, hor)
        painter.setPen(pg.mkPen((165, 42, 42), width=2.0*self.x_scale))
        painter.drawLine(x_scale*(x+x_max), hor, x_scale*(x+x_max-l*sin(theta)), hor-h_scale*(l*cos(theta)))
        painter.setPen(pg.mkPen('b'))
        painter.setBrush(pg.mkBrush('b'))
        painter.drawRect(x_scale*(x+x_max)-c_w/2, hor-c_h/2, c_w, c_h)
        painter.setPen(pg.mkPen('r'))
        painter.setBrush(pg.mkBrush('r'))
        painter.drawEllipse(x_scale*(x+x_max-l*sin(theta)-r/2), hor-h_scale*(l*cos(theta)+r/2), r*x_scale, r*h_scale)
        painter.setPen(pg.mkPen('k'))
        for i in np.arange(-x_max, x_max, x_max/10):
            painter.drawText((i+x_max)*x_scale, image_h-10, str(int(i)))
        for i in np.arange(h_min, h_max, (h_max-h_min)/10):
            painter.drawText(0, image_h-(int(i)-h_min)*h_scale, str(int(i)))

    # Equation of the mechanics of the pendulum
    def solve_equation(self, F):
        l, m, M = self.l, self.m, self.M
        g = 9.81
        a11 = M+m
        a12 = -m*l*cos(self.theta)
        b1 = F-m*l*self.dtheta**2*sin(self.theta)
        a21 = -cos(self.theta)
        a22 = l
        b2 = g*sin(self.theta)
        a = np.array([[a11, a12], [a21, a22]])
        b = np.array([b1, b2])
        sol = np.linalg.solve(a, b)
        return sol[0], sol[1]

    # Numerical integration of acceleration
    def count_state_params(self, F, dt=0.001):
        ddx, ddtheta = self.solve_equation(F)
        self.dx += ddx*dt
        self.x += self.dx*dt
        self.dtheta += ddtheta*dt
        self.theta += self.dtheta*dt
        self.theta = arctan2(sin(self.theta), cos(self.theta))

    # Simulation run
    # The sandbox variable tells whether the simulation should be aborted in the event of a control failure -
    # (Deflect too much or pendulum too horizontal)
    def run(self, sandbox=false, frameskip=20):
        self.sandbox = sandbox
        self.frameskip = frameskip
        self.init_image()
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.single_loop_run)
        timer.start(1)

    # n - times calculation of the next state of the system
    def single_loop_run(self):
        for i in range(self.frameskip+1):
            dis=next(self.disruption, 0)
            control = self.fuzzy_control(self.x, self.theta, self.dx, self.dtheta)
            F = dis+control
            self.count_state_params(F)
            if not self.sandbox:
                if self.x < -self.x_max or self.x > self.x_max or np.abs(self.theta) > np.pi/3:
                    exit(1)
        self.update()
        print(f'theta:{self.theta}')
        print(f'dtheta:{self.dtheta}')
        print(f'x:{self.x}')
        print(f'dx:{self.dx}')

    # Fuzzy regulator
    def init_blurring(self, x, dx, theta, dtheta):
        # 1. Fuzzying
        global S, L, R, ML, MR, SL, MSL, B, SR, MSR, VD, LL, MLL, RR, MRR, VU, LMA, LA, AN, RA, RMA, FML, FL, NN, FR, FMR, FF

        a = 0
        b = 10
        c = 50
        d = 0
        e = 5
        f = 20

        # Membership functions for road x
        # S - stop(reached 0)
        # L - cart on the left
        # ML - cart more on the right
        # R - cart on the right
        # MR - cart more on the right

    # S(x)
        if x <= -b or x >= b:
            S = 0
        if a > x > -b:
            S = (x + b) / (a + b)
        if x == a:
            S = 1
        if b > x > a:
            S = (b - x) / (b - a)
        print(f'S={S}')

    # L(x)
        if x <= -c or x >= a:
            L = 0
        if -b > x > -c:
            L = (x + c) / (-b + c)
        if x == -b:
            L = 1
        if a > x > -b:
            L = (a - x) / (a + b)
        print(f'L={L}')

    # R(x)
        if x <= a or x >= c:
            R = 0
        if b > x > a:
            R = (x - a) / (b - a)
        if x == b:
            R = 1
        if c > x > b:
            R = (c - x) / ( c - b)
        print(f'R={R}')

    # ML
        if x >= -b:
            ML = 0
        if -c < x < -b:
            ML = (-b - x) / (-b + c)
        if x <= -c:
            ML = 1
        print(f'ML={ML}')

    # MR
        if x <= b:
            MR = 0
        if c > x > b:
            MR = (x - b) / (c - b)
        if x >= c:
            MR = 1
        print(f'MR={MR}')

        # Membership functions for speed dx
        # MSL - higher speed left
        # SL - speed left
        # B - cart have speed 0
        # SR - speed right
        # MSR - higher speed right

    # MSL
        if dx >= -e:
            MSL = 0
        if -e < dx < -f:
            MSL = (-e + dx) / (-e + f)
        if dx <= -f:
            MSL = 1
        print(f'MSL={MSL}')
    # SL
        if dx <= -f or dx >= d:
            SL = 0
        if -e > dx > -f:
            SL = (dx + f) / (-e + f)
        if dx == -e:
            SL = 1
        if d > dx > -e:
            SL = (d - dx) / (d + e)
        print(f'SL={SL}')
    # B
        if -e >= dx or dx >= e:
            B = 0
        if d > dx > -e:
            B = (dx + e) / (d + e)
        if dx == d:
            B = 1
        if e > dx > d:
            B = (e - dx) / (e - d)
        print(f'B={B}')

    # SR
        if dx <= d or dx >= f:
            SR = 0
        if e > dx > d:
            SR = (dx - d) / (e - d)
        if dx == e:
            SR = 1
        if f > dx > e:
            SR = (f - dx) / (f - e)
        print(f'SR={SR}')
        print(f'x:{self.x}')
        print(f'dx:{self.dx}')

    # MSR
        if dx <= e:
            MSR = 0
        if f > dx > e:
            MSR = (dx - e) / (f - e)
        if dx >= f:
            MSR = 1
        print(f'MSR={MSR}')

        # Membership functions for angular position of pendulum (theta)
        # VD - vertically down
        # LL - in left part
        # RR - in right part
        # VU - vertically up
        k = np.pi/17
        n = np.pi/12
        t = np.pi
        r = 0

    # VD(theta)
        if n >= theta >= -n:
            VD = 0
        if -n > theta > -t:
            VD = (-n - theta) / (-n + t)
        if theta == -t or theta == t:
            VD = 1
        if n < theta < t:
            VD = (theta - n) / (t - n)
        print(f'VD={VD}')

    # MRR
        if theta <= -t or theta >= -k:
            MRR = 0
        if -n > theta > -t:
            MRR = (theta + t) / (-n + t)
        if theta == -n:
            MRR = 1
        if -k > theta > -n:
            MRR = (-k - theta) / (-k + n)
        print(f'MRR={MRR}')

    # LL(theta)
        if theta <= r or theta >= n:
            LL = 0
        if k > theta > r:
            LL = (theta - r) / (k - r)
        if theta == r:
            LL = 1
        if k < theta < n:
            LL = (n - theta) / (n - k)
        print(f'LL={LL}')

    # VU(theta)
        if theta >= k or theta <= -k:
            VU = 0
        if r < theta < k:
            VU = (k - theta) / (k - r)
        if theta == r:
            VU = 1
        if -k < theta < r:
            VU = (r - theta) / (r + k)
        print(f'VU={VU}')

    # RR(theta)
        if theta >= r or theta <= -n:
            RR = 0
        if -k > theta > -n:
            RR = (theta + n) / (-k + n)
        if theta == -k:
            RR = 1
        if r > theta > -k:
            RR = (theta + k) / (r + k)
        print(f'RR={RR}')
    # MLL
        if theta <= k or theta >= t:
            MLL = 0
        if n > theta > k:
            MLL = (theta - k) / (n - k)
        if theta == n:
            MLL = 1
        if t > theta > n:
            MLL = (t - theta) / (t - n)
        print(f'MLL={MLL}')

        # Membership functions for angular speed (dtheta)
        # LMA - higher acceleration left
        # LA - acceleration left
        # AN - no acceleration
        # RA - acceleration right
        # RMA - higher acceleration right
        s = 3.482
        t = 1.236
        w = 0

    # LMA(dtheta)
        if dtheta < t:
            LMA = 0
        if t < dtheta < s:
            LMA = (t - dtheta) / (t - s)
        if dtheta >= s:
            LMA = 1
        print(f'LMA={LMA}')

    # LA(dtheta)
        if dtheta <= w or dtheta >= s:
            LA = 0
        if s > dtheta > t:
            LA = (dtheta - s) / (t - s)
        if dtheta == t:
            LA = 1
        if w < dtheta < t:
            LA = (w - dtheta) / (w - t)
        print(f'LA={LA}')

    # AN(dtheta)
        if dtheta > t or dtheta < -t:
            AN = 0
        if t > dtheta > w:
            AN = (dtheta - t) / (w - t)
        if dtheta == w:
            AN = 1
        if w > dtheta > -t:
            AN = (-t - dtheta) / (-t - w)
        print(f'AN={AN}')

    # RA(dtheta)
        if dtheta >= w or dtheta <= -s:
            RA = 0
        if w > dtheta > -t:
            RA = (dtheta - w) / (-t - w)
        if dtheta == -t:
            RA = 1
        if -t > dtheta > -s:
            RA = (-s - dtheta) / (-s + t)
        print(f'RA={RA}')

    # RMA(dtheta)
        if dtheta > -t:
            RMA = 0
        if -t > dtheta > -s:
            RMA = (dtheta + t) / (-s + t)
        if dtheta <= -s:
            RMA = 1
        print(f'RMA={RMA}')

        # Membership functions for force applied to cart F(x,dx,theta,dtheta)
        # FML - higher force to the left
        # FL - force to the left
        # NN - no force applied
        # FR - force to the right
        # FMR - higher force to the right
        y = 700
        z = 60
        zz = 0
        FF = float(15*dx)
        print(f'FF={FF}')

    # FML(F)
        if FF >= -z:
            FML = 0
        if -z > FF > -y:
            FML = (-z - FF) / (-z + y)
        if FF <= -y:
            FML = 1
        print(f'FML={FML}')

    # FL(F)
        if FF <= -y or FF > zz:
            FL = 0
        if -z > FF > -y:
            FL = (FF + y) / (-z + y)
        if FF == -z:
            FL = 1
        if zz >= FF > -z:
            FL = (zz - FF) / (zz + z)
        print(f'FL={FL}')


    # NN(F)
        if FF < -z or FF >= z:
            NN = 0
        if zz > FF > -z:
            NN = (FF + z) / (zz + z)
        if FF == zz:
            NN = 1
        if z > FF > zz:
            NN = (z - FF) / (z - FF)
        print(f'NN={NN}')


    # FR(F)
        if FF <= zz or FF >= y:
            FR = 0
        if z > FF > zz:
            FR = (FF - zz) / (z - zz)
        if FF == z:
            FR = 1
        if y > FF > z:
            FR = (y - FF) / (y - z)
        print(f'FR={FR}')

    # FMR(F)
        if FF <= z:
            FMR = 0
        if y > FF > z:
            FMR = (FF - z) / (y - z)
        if FF >= y:
            FMR = 1
        print(f'FMR={FMR}')

        return

    def suma(self, a, b, c):
        suma = max(a, b, c)
        return suma

    def dop(self, a):
        dop = 1 - a
        return dop

    def ilo(self, a, b):
        ilo = min(a, b)
        return ilo


    def fuzzy_control(self, x, theta, dx, dtheta):
        global Z
        InvertedPendulum.init_blurring(self, x, dx, theta, dtheta)
#NN
        u1 = InvertedPendulum.ilo(self, VU, AN)
        u12 = InvertedPendulum.ilo(self, u1, S)
        u2 = InvertedPendulum.ilo(self, RR, LA)
        u3 = InvertedPendulum.ilo(self, LL, RA)
        u123 = InvertedPendulum.suma(self, u12, u2, u3)
#FL
        u4 = InvertedPendulum.suma(self, R, MR, 0)
        u5 = InvertedPendulum.suma(self, SR, MSR, 0)
        u45 = InvertedPendulum.ilo(self, u4, u5)
        u45n = InvertedPendulum.dop(self, u45)
        u6 = InvertedPendulum.ilo(self, u45n, LL)

#FR
        u7 = InvertedPendulum.suma(self, L, ML, 0)
        u8 = InvertedPendulum.suma(self, SL, MSL, 0)
        u78 = InvertedPendulum.ilo(self, u7, u8)
        u78n = InvertedPendulum.dop(self, u78)
        u9 = InvertedPendulum.ilo(self, u78n, RR)

#FML
        u10 = MLL
#FMR
        u11 = MRR
#FL
        u12 = InvertedPendulum.ilo(self, LA, LA)
        u13 = InvertedPendulum.ilo(self, VD, u12)
#FR
        u14 = InvertedPendulum.ilo(self, RA, RA)
        u15 = InvertedPendulum.ilo(self, VD, u14)

#Aggregation
        ufl = InvertedPendulum.suma(self, u6, u13, 0)
        ufr = InvertedPendulum.suma(self, u9, u15, 0)

#Sharpening
#Height method
        Z1 = u123 + ufl + ufr + u10 + u11
        Z = (u123*0 + ufl*(-60) + ufr*60 + u10*(-700) + u11*700) / (Z1)
        return Z

if __name__ == '__main__':
    app = QtGui.QApplication(argv)
    if len(argv)>1:
        ip = InvertedPendulum(f_name=argv[1])
    else:
        ip = InvertedPendulum(x0=90, dx0=0, theta0=0, dtheta0=0.1, ih=800, iw=1000, h_min=-80, h_max=80)
    ip.run(sandbox=True)
    exit(app.exec_())
