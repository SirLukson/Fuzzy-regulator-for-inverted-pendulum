#include <QtWidgets>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>

using namespace std;

class InvertedPendulum : public QWidget{
private:
    double M, m, l, x0, theta0, dx0, dtheta0;
    double image_w, image_h, x_max, h_min, h_max;
    bool dis_cyc;
    vector<double> disruption;
    vector<double>::iterator dis_it;
    bool sandbox;
    double hor, c_w, c_h, r;
    double x, theta, dx, dtheta;
    double h_scale, x_scale;

    int frameskip;
    void init_image();
    void update_image(double x, double theta);
    void solve_equation(double &ddx, double &ddtheta, double F);
    void count_state_params(double F, double dt);
    double fuzzy_control();
public:
    InvertedPendulum(double, double, double, double, double, double, double,
                     int, int, int, int, int,
                     bool,
                     vector<double>);
    InvertedPendulum(string);
    InvertedPendulum(const InvertedPendulum&) = default;
    ~InvertedPendulum()=default;
    void run(bool sandbox, int frameskip);
    void single_loop_run();
protected:
    void paintEvent(QPaintEvent *event);
};

void InvertedPendulum::paintEvent(QPaintEvent *event) {
    QPainter painter(this);
    painter.setPen(QPen(QColor(165, 42, 42), 2.0*x_scale));
    painter.drawLine(x_scale*(x+x_max), hor, x_scale*(x+x_max-l*sin(theta)), hor-h_scale*(l*cos(theta)));
    painter.setPen(QPen(Qt::black, 2.0*h_scale));
    painter.drawLine(0, hor, image_w, hor);
    painter.setPen(QPen(Qt::blue));
    painter.setBrush(QBrush(Qt::blue));
    painter.drawRect(x_scale*(x+x_max)-c_w/2, hor-c_h/2, c_w, c_h);
    painter.setPen(QPen(Qt::red));
    painter.setBrush(QBrush(Qt::red));
    painter.drawEllipse(x_scale*(x+x_max-l*sin(theta)-r/2), hor-h_scale*(l*cos(theta)+r/2), r*x_scale, r*h_scale);
    painter.setPen(QPen(Qt::black));
    for (double i = -x_max; i<=x_max; i+= x_max/10) {
        int x_lab = int(i);
        painter.drawText((i+x_max)*x_scale, image_h-10, QString::number(x_lab));
    }
    for (double i=h_min; i<=h_max; i+= (h_max-h_min)/10) {
        int h_lab = int(i);
        painter.drawText(0, image_h-(h_lab-h_min)*h_scale, QString::number(h_lab));
    }
}

InvertedPendulum::InvertedPendulum(double M=10, double m=5, double l=50, double x0=0, double theta0=0, double dx0=0, double dtheta0=0,
                                   int iw=1000, int ih=500, int x_max=100, int h_min=0, int h_max=100,
                                   bool dis_cyc=true,
                                   vector<double> disruption = vector<double>{0}):
    M(M), m(m), l(l), x0(x0), theta0(theta0), dx0(dx0), dtheta0(dtheta0),
    image_w(iw), image_h(ih), x_max(x_max), h_min(h_min), h_max(h_max),
    dis_cyc(dis_cyc), disruption(disruption) {};

InvertedPendulum::InvertedPendulum(string f_name) {
    fstream f_stream;
    f_stream.open(f_name, ios::in);
    string line;
    int i=0;
    while(getline(f_stream, line)) {
        switch (i) {
        case 0: {
            istringstream iss(line);
            vector<string> vars{istream_iterator<string>{iss},
                                istream_iterator<string>{}};
            M = stod(vars[0]);
            m = stod(vars[1]);
            l = stod(vars[2]);
            x0 = stod(vars[3]);
            theta0 = stod(vars[4]);
            dx0 = stod(vars[5]);
            dtheta0 = stod(vars[6]);
            image_w = stoi(vars[7]);
            image_h = stoi(vars[8]);
            x_max = stoi(vars[9]);
            h_min = stoi(vars[10]);
            h_max = stoi(vars[11]);
            break;
        }
        case 1: {
            if (line=="1")
                dis_cyc = true;
            break;
        }
        case 2: {
            istringstream iss(line);
            disruption = vector<double>{istream_iterator<double>(iss),
                    istream_iterator<double>()};
            break;
        }
        }
        i++;
    }
}

void InvertedPendulum::init_image() {
    h_scale = image_h/(h_max-h_min);
    x_scale = image_w/(2*x_max);
    hor = (h_max-10)*h_scale;
    c_w = 16*x_scale;
    c_h = 8*h_scale;
    r = 8;
    x = x0;
    theta = theta0;
    dx = dx0;
    dtheta = dtheta0;
    setFixedSize(image_w, image_h);
    show();
    setWindowTitle("Inverted Pendulum");
    setPalette(Qt::white);
    update();
}

void InvertedPendulum::solve_equation(double &ddx, double &ddtheta, double F) {
    double g = 9.81;
    double a11 = M+m;
    double a12 = -m*l*cos(theta);
    double b1 = F-m*l*pow(dtheta, 2)*sin(theta);
    double a21 = -cos(theta);
    double a22 = l;
    double b2 =g*sin(theta);
    ddtheta = (b1/a11-b2/a21)/(a12/a11-a22/a21);
    ddx = (b1-a12*ddtheta)/a11;
}

void InvertedPendulum::count_state_params(double F, double dt=0.001) {
    double ddx, ddtheta;
    solve_equation(ddx, ddtheta, F);
    dx += ddx*dt;
    x += dx*dt;
    dtheta += ddtheta*dt;
    theta += dtheta*dt;
    theta = atan2(sin(theta), cos(theta));
}

void InvertedPendulum::single_loop_run() {
    for (int i=0; i<frameskip+1; i++) {
        double control = fuzzy_control();
        double F = control + *dis_it;
        dis_it ++;
        if (dis_it==disruption.end()) {
            if (dis_cyc) {
                dis_it = disruption.begin();
            } else {
                dis_cyc = true;
                disruption = vector<double>{0};
            }
        }
        count_state_params(F);
        if (!sandbox && (x<-x_max || x>x_max || fabs(theta) > 3.14159265/3)) {
            exit(1);
        }
    }
    update();
}
void InvertedPendulum::run(bool sandbox, int frameskip=20) {
    this->sandbox = sandbox;
    this->frameskip = frameskip;
    init_image();
    dis_it = disruption.begin();
    QTimer *m_timer = new QTimer(this);
    connect(m_timer, &QTimer::timeout, this, &InvertedPendulum::single_loop_run);
    m_timer->start(1);
}

double InvertedPendulum::fuzzy_control() {
    return 0;
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    InvertedPendulum *inv_pend;
    if (argc>1)
        inv_pend = new InvertedPendulum(argv[1]);
    else
        inv_pend = new InvertedPendulum(10, 5, 50, 90, 0, 0, 0.1, 1000, 800, 100, -80, 80);
    inv_pend->run(true);
    return app.exec();
}
