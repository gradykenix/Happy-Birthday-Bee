'''
ascii art
darwinian algorithm
bee :)
flower arrangement
liquid-drop model

use home-built templates to put together the bouquet
then use colour theory to colorize the bouquet TICK
then, use atomic nucleus structuring to make the bouquet spherical TICK
then, display it in 3D, rotating, with ascii (95%)
'''

'''
update ideas:
personally made bouquets (cause these look like hot shit lol)
'''

# assorted libraries
import random
import math
import time

# assorted constants
WIDTH = 74
HEIGHT = 41
THETA = 2*math.pi/64
PHI = 0 # seriously broken!!
WAIT_TIME = 0.02

MIN_VASE_RADIUS = 10
MAX_VASE_RADIUS = 15
VASE_HEIGHT = 15

RADIUS = 80 # modelling constant

REID_MIN = 0.8452 # for the nuclear model
REID_STEPS = 800 # number of rounds of physics sim
PACK_DIST = 20
REID_DT = 0.01

MIN_PETALS = 70
MAX_PETALS = 100

LIGHTEN = 45

print("ASCII BOUQUET GENERATOR")
time.sleep(1)
print("for Bee -- happy birthday!!")
time.sleep(2)
print('''Feel free to read the docs or the code docstring
to learn more about what's going on behind the scenes :)''')
time.sleep(2)
print("Time to get started!")
time.sleep(1.5)
y = input('''
First things first -- do you prefer the blob-mode
( not very buggy, but low detail, only using the 'o' character)
or the high-res mode ( horrifically buggy, your odds of
getting something recognizable are ~1/10**9, uses a variety
of flower templates )?
 [1] blob mode
 [2] high res mode

 :: ''')
while y not in ['1','2']:
    y = input('''
Could you say that again? :: ''')
if y == '1':
    MODE = 'BLOB'
else:
    MODE = 'HIGH-RES'

TRUECOLOR = True # expect periodic jumps if False
# CHECK IF YOUR TERMINAL CAN HANDLE TRUECOLOR IF YOU ARE HAVING ISSUES !!!
FLOWER_VARIANCE = 0
color_wheel = [(255,0,0), (255,125,0), (255,255,0), (125,255,0), (0,255,0), (0,255,125),
               (0,255,255), (0,125,255), (0,0,255), (125,0,255), (255,0,255), (255,0,125)] # filled in at vec

NUM_FLOWERS = 50

# assorted functions
def rand(a,b):
    x = random.random()
    # linear transform 0 to a and 1 to b
    # y = mx + c s.t. c = a and m = b-a
    return (b-a)*x + a          

def char_perception(note, view_pos):
    # theta is x-y plane rotation 0 - 360, phi is z-axis rotation 0 - 180

    if (note.pos - view_pos).l >= (400*400):
        return '. ' # too far away
    # first obtain effective angulations
    pos = note.pos - view_pos
    p_theta, p_phi = pos.get_theta(), pos.get_phi()
    
    perc = vec(1,0,0).rot(note.theta - p_theta, note.phi - p_phi)
    perc_theta, perc_phi = perc.get_theta(), perc.get_phi()
    
    if note.char == 'o':
        # meant to signify a little orb
        return 'o '
    if note.char == 'O':
        # meant to signify a big orb
        return 'O '
    if note.char == '.':
        # meant to signify a point
        return '. '
    if note.char == '^':
        # meant to signify a up-down bent stroke
        # can be ^ v - \ / |
        # naturally xy-embedded, pointing away from origin
        if (-math.pi/6 < perc_phi) and (perc_phi < math.pi/6):
            return '- '
        if perc_phi > 2*math.pi/3:
            if 3*math.pi/8 < perc_theta < 5*math.pi/8:
                return '| '
            if 11*math.pi/8 < perc_theta < 15*math.pi/8:
                return '| '
            return 'v '
        if perc_phi < -2*math.pi/3:
            if 3*math.pi/8 < perc_theta < 5*math.pi/8:
                return '| '
            if 11*math.pi/8 < perc_theta < 15*math.pi/8:
                return '| '
            return '^ '
        if perc_phi > 0:
            if math.pi/8 < perc_theta < 3*math.pi/8:
                return '\\ '
            if 5*math.pi/8 < perc_theta < 7*math.pi/8:
                return '/ '
            if 3*math.pi/8 < perc_theta < 5*math.pi/8:
                return '| '
            if 11*math.pi/8 < perc_theta < 15*math.pi/8:
                return '| '
            return 'v '
        else:
            if math.pi/8 < perc_theta < 3*math.pi/8:
                return '/ '
            if 5*math.pi/8 < perc_theta < 7*math.pi/8:
                return '\\ '
            if 3*math.pi/8 < perc_theta < 5*math.pi/8:
                return '| '
            if 11*math.pi/8 < perc_theta < 15*math.pi/8:
                return '| '
            return '^ '
            
    if note.char == '-':
        # meant to signify a linear stroke length 1
        # can be | \ / - .
        # natural state: along x axis, facing origin
        if (-math.pi/5 < perc_phi) and (perc_phi < math.pi/5):
            if perc_theta < math.pi / 4:
                return '. '
            if math.pi*3/4 < perc_theta < math.pi*5/4:
                return '. '
            if math.pi*7/4 < perc_theta:
                return '. '
            return '- '
        if perc_theta <= math.pi/6:
            return '| '
        if math.pi*5/6 <= perc_theta <= math.pi*7/6:
            return '| '
        if math.pi*11/6 <= perc_theta:
            return '| '
        if perc_phi > 0:
            if perc_theta < math.pi:
                return '/ '
            return '\\ '
        if perc_theta < math.pi:
            return '\\ '
        return '/ '
    return note.char

def V_Reid(x):
    '''The Reid potential -- not really the liquid-drop model :)
Minimum value at 0.8452, called REID_MIN.'''
    m = 0.7
    k = 0.72
    t1 = -10.463*(math.e**(-m*x))/(m*x)
    t2 = -1650.6*(math.e**(-4*m*x))/(m*x)
    t3 = 6484.2*(math.e**(-7*m*x))/(m*x)
    return t1+t2+t3+k*x # color confinement +k*x

def F_Reid(x):
    '''The derivative of the Reid potential, i.e. the strong force.'''
    m = 0.7
    k = 0.72
    t1 = -10.463*(math.e**(-m*x))/(m*x)
    t1 = -(m+1/x)*t1
    t2 = -1650.6*(math.e**(-4*m*x))/(m*x)
    t2 = -(4*m+1/x)*t2
    t3 = 6484.2*(math.e**(-7*m*x))/(m*x)
    t3 = -(7*m+1/x)*t3
    return -(t1+t2+t3+k)
        

# vector class WIP
class vec:
    '''The vector class exists to make 3D and 2D
object manipulation easier. '''
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
        self.l = x*x + y*y + z*z
        
    def dot(self, other):
        x = self.x*other.x
        y = self.y*other.y
        z = self.z*other.z
        return x + y + z
    
    def __mul__(self, scalar):
        x = self.x*scalar
        y = self.y*scalar
        z = self.z*scalar
        return vec(x, y, z)
    
    def color_256(self):
        return color_table_256[self.x][self.y][self.z] # precomputed
    
    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        z = self.z + other.z
        return vec(x, y, z)
    
    def display(self):
        print(self.x, self.y, self.z)
        
    def in_bounds(self):
        return self.l <= RADIUS**2
    
    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        z = self.z - other.z
        return vec(x, y, z)
    
    def cross(self, other):
        x = self.y*other.z - self.z*other.y
        y = self.z*other.x - self.x*other.z
        z = self.x*other.y - self.y*other.x
        return vec(x, y, z)
    
    def cpy(self):
        return vec(self.x, self.y, self.z)
    
    def rot(self, theta, phi):
        ct = math.cos(theta)
        st = math.sin(theta)
        cp = math.cos(phi)
        sp = math.sin(phi)
        x_prime = self.x*ct - self.y*st
        y_prime = self.x*st + self.y*ct
        r = (self.x**2 + self.y**2)**0.5
        r_prime = cp*r - self.z*sp
        z_prime = sp*r + self.z*cp
        x_prime *= r_prime/r if r else 1
        y_prime *= r_prime/r if r else 1
        return vec(x_prime, y_prime, z_prime)
    
    def get_theta(self):
        if self.x == 0:
            if self.y >= 0:
                return math.pi / 2
            return 3*math.pi/2
        if self.x > 0 and self.y >= 0:
            return math.atan(self.y / self.x)
        if self.x > 0:
            return 2*math.pi + math.atan(self.y / self.x)
        return math.pi + math.atan(self.y / self.x)

    def get_phi(self):
        r = (self.x**2 + self.y**2)**0.5
        if r == 0:
            if self.z >= 0:
                return math.pi / 2
            return -math.pi/2
        return math.atan(self.z / r)

    def max(self):
        return max(self.x, self.y, self.z)

    def proper(self):
        return (0 <= self.x < 256) and (0 <= self.y < 256) and (0 <= self.z < 256)

    def unit(self):
        return self*(1/self.l**0.5)

    def __eq__(self, other):
        return (self.x == other.x) and (self.y == other.y) and (self.z == other.z)

color_wheel = [vec(*i) for i in color_wheel]
        
if not TRUECOLOR:
    color_table_256 = [[[0 for r in range(256)] for g in range(256)] for b in range(256)]
    colors_256 = [0 for i in range(256)] # both filled under vec
    # system colours
    colors_256 = [vec(0,0,0), vec(170,0,0), vec(0,170,0), vec(170,170,0), vec(0,0,170), 
                  vec(170,0,170), vec(0,170,170), vec(170,170,170), vec(85,85,85),
                  vec(255,85,85), vec(85,255,85), vec(255,255,85), vec(85,85,255),
                  vec(255,85,255), vec(85,255,255), vec(255,255,255)]
    # 6x6x6 colour cube!
    for r in range(6):
        for g in range(6):
            for b in range(6):
                colors_256.append(vec(51*r, 51*g, 51*b))
    # grayscale ladder
    for i in range(24):
        colors_256.append(vec((256*i)//24, (256*i)//24, (256*i)//24))

    # color_table_256 -- nearest color to each RGB integer
    for r in range(256):
        for g in range(256):
            for b in range(256):
                color_table_256[r][g][b] = min([(colors_256[i].dist(vec(r,g,b)), i) for i in range(256)])[1]

# petal class
class Petal:
    '''Petals are individual one-ASCII-char objects,
initially white and unoriented by default. '''
    def __init__(self, pos, char):
        self.pos = pos
        self.color = vec(255, 255, 255) # white
        self.og_color = vec(255, 255, 255)
        self.char = char
        self.theta = 0
        self.phi = 0

    def colorize(self, color):
        self.color = color

    def set_og_color(self, color):
        self.og_color = color

    def cpy(self):
        p = Petal(self.pos.cpy(), self.char)
        p.colorize(self.color.cpy())
        p.set_og_color(self.og_color.cpy())
        p.theta = self.theta
        p.phi = self.phi
        return p

    def rot(self, theta, phi):
        self.pos = self.pos.rot(theta, phi)
        self.theta += theta
        self.phi += phi
        if self.phi > math.pi/2:
            self.phi = math.pi - self.phi
            self.theta += math.pi
            self.theta = self.theta % (2*math.pi)
        elif self.phi < -math.pi/2:
            self.phi = -math.pi - self.phi
            self.theta += math.pi
            self.theta = self.theta % (2*math.pi)

    def __lt__(self, other): # necessary for sorting
        return self

#flower class
class Flower:
    '''Flowers are collections of petals!'''
    def __init__(self):
        self.petals = []
        self.pos = vec(0,0,0) # origin
        self.pointing = vec(0,0,0) # initially

    def new_flower(self, template):
        if MODE == 'BLOB':
            center = vec(*[rand(-RADIUS//2, RADIUS//2) for i in range(3)])
            self.petals = [Petal(center, 'O')]
            self.pos = center
            for i in range(random.randint(MIN_PETALS, MAX_PETALS)):
                pos = random.choice(self.petals).pos
                plus = vec(*[rand(-1,1) for i in range(3)])
                if pos != self.pos:
                    plus += (pos - self.pos).unit()
                self.petals.append(Petal(pos+plus, random.choice(['o', 'o', 'O'])))
                self.petals[-1].theta = rand(0, 2*math.pi)
                self.petals[-1].phi = rand(-math.pi, math.pi)
            self.colorize(vec(*[rand(150,255) for i in range(3)]))
            return self

        # initialize
        center = vec(*[rand(0, RADIUS//4), rand(-RADIUS//4, RADIUS//4), rand(-RADIUS//4, RADIUS//4)])
        dist = center.l**0.5
        template = random.choice(TEMPLATES).cpy() if (template is None) else template
        self.pos = center
        self.petals = []
        theta = center.get_theta()
        xy_matrix_r1 = vec(math.cos(theta), -math.sin(theta), 0)
        xy_matrix_r2 = vec(math.sin(theta), math.cos(theta), 0)
        phi = center.get_phi()
        rz_matrix_r2 = vec(math.sin(phi), math.cos(phi), 0)

        # rotate template to center
        for petal in template.petals:
            x = xy_matrix_r1.dot(petal.pos+vec(dist-1,0,0))
            y = xy_matrix_r2.dot(petal.pos+vec(dist-1,0,0))
            z = rz_matrix_r2.dot(vec(x, y, 0))
            petal.pos = vec(x,y,z)
            petal.theta -= theta
            if petal.theta < 0:
                petal.theta += 2*math.pi
            petal.phi -= phi
            if petal.phi > math.pi:
                petal.phi = math.pi - petal.phi
                petal.theta += math.pi
                petal.theta = petal.theta % (2*math.pi)
            elif petal.phi < -math.pi:
                petal.phi = -math.pi - petal.phi
                petal.theta += math.pi
                petal.theta = petal.theta % (2*math.pi)
            self.petals.append(petal)
        
        return self

    def cpy(self):
        new_petals = [p.cpy() for p in self.petals]
        new_cpy = Flower()
        new_cpy.petals = new_petals
        new_cpy.n = len(new_petals)
        new_cpy.pos = self.pos.cpy()
        return new_cpy

    def colorize(self, color):
        # introduce a new colour scheme
        for petal in self.petals:
            petal.set_og_color(color * rand(LIGHTEN/100, 1))
            petal.colorize(petal.og_color.cpy())

# vase class
class Vase:
    '''Vases are simple and rotationally symmetric ASCII objects;
later, amphorae (asymmetric) or other species might be introduced. '''
    def __init__(self):
        # randomly generate an entire vase
        self.edges = []
        radius = rand(MIN_VASE_RADIUS, MAX_VASE_RADIUS)
        radii = [radius]
        for i in range(VASE_HEIGHT):
            if radius == MIN_VASE_RADIUS:
                radius = random.choice([radius+1, radius])
            elif radius == MAX_VASE_RADIUS:
                radius = random.choice([radius, radius-1])
            else:
                radius = random.choice([radius+1, radius, radius-1])
            radii.append(radius)
        for i in range(5):
            radii.append(radii[-1]-1)
        h = 10
        #rim
        for i in range(0, 360):
            k = i*math.pi/180
            rim = Petal(vec(radii[0]*math.cos(k), radii[0]*math.sin(k), h), '-')
            rim.theta = math.pi + (k)
            self.edges.append(rim)
        
        for r in range(1, len(radii)):
            petal_left = Petal(vec(0, radii[r], h), ['| ','/ ','\\ '][int(radii[r]-radii[r-1])])
            petal_right = Petal(vec(0, -radii[r], h), ['| ','\\ ','/ '][int(radii[r]-radii[r-1])])
            self.edges.append(petal_left)
            self.edges.append(petal_right)
            h += 1
        #base rim
        for i in range(0, 90):
            k = i*math.pi/180
            rim = Petal(vec(radii[0]*math.cos(k), radii[0]*math.sin(k), h), '-')
            rim.theta = math.pi + (k)
            self.edges.append(rim)
        for i in range(270, 360):
            k = i*math.pi/180
            rim = Petal(vec(radii[0]*math.cos(k), radii[0]*math.sin(k), h), '-')
            rim.theta = math.pi + (k)
            self.edges.append(rim)
        self.radii = radii
    # note vases are silhouettes, made only of edges
    # when a vase rotates the edge does not pass in front!!
    # they have no interior content, no patterns, no handles

# bouquet class
class Bouquet:
    '''Bouquets comprise a vase and a set of flowers, which
can be arranged and coloured into a display-ready outcome. '''
    def __init__(self, flowers, vase):
        self.vase = vase
        self.flowers = flowers
        self.n = len(flowers)

    def bee_principle(self):
        # put down-pointing flowers down,
        # and up-pointing flowers up;
        # run before liquid_drop
        pass
    
    def liquid_drop_step(self):
        # arrange into rough sphere
        updated_flowers = []
        alpha = REID_MIN / PACK_DIST
        for f in range(self.n):
            flower = self.flowers[f]
            resultant = vec(0,0,0)
            for other_flower in self.flowers:
                if other_flower.pos != flower.pos:
                    diff = flower.pos - other_flower.pos
                    resultant += diff.unit()*min(F_Reid(alpha * (diff.l**0.5)), 100) # to avoid banishment
            # F = a; all masses are assumed to be units
            updated_flower = flower.cpy()
            delt = resultant*(REID_DT**2)
            updated_flower.pos += delt
            for petal in updated_flower.petals:
                petal.pos += delt
            updated_flowers.append(updated_flower)
        
        self.flowers = updated_flowers
        
    def colorize(self, color_scheme):
        # use colour theory
        if color_scheme == None:
            color_scheme = random.choice(['triadic', 'analogous', 'monochromatic', 'complementary', 'rainbow!'])
        L = len(color_wheel)
        i = random.randint(0, L-1)
        if color_scheme == 'triadic':
            palette = [color_wheel[i], color_wheel[(i+(L//3))%L], color_wheel[(i+((2*L)//3))%L]]
        elif color_scheme == 'analogous':
            palette = [color_wheel[i], color_wheel[(i-1)%L], color_wheel[(i+1)%L]]
        elif color_scheme == 'monochromatic':
            palette = [color_wheel[i]]
        elif color_scheme == 'complementary':
            palette = [color_wheel[i], color_wheel[i], color_wheel[(i+(L//2))%L]]
        elif color_scheme == 'rainbow!':
            palette = color_wheel
        else:
            palette = [vec(*i) for i in color_scheme]
        self.color_scheme = color_scheme
        # now apply to make the colors!
        for flower in self.flowers:
            base_color = random.choice(palette)
            for petal in flower.petals:
                og_color = (base_color + vec(*[random.randint(-FLOWER_VARIANCE, FLOWER_VARIANCE) for i in range(3)]))*(rand(LIGHTEN/100, 1))
                petal.set_og_color(og_color)
                petal.colorize(og_color)

    def rotate(self, theta, phi):
        # rotate flowers
        for flower in self.flowers:
            for petal in flower.petals:
                petal.rot(theta, 0)

    def shade(self, vantage):
        norm = vantage*(RADIUS / (vantage.l**0.5)) # this is being computed many times -- not necessary?
        norm_dist = (norm*(-1) + vantage).l
        norm_dist = vantage.l
        for flower in bouquet.flowers:
            for petal in flower.petals:
                dist = (petal.pos*(-1) + vantage).l
                petal.colorize(petal.og_color*(norm_dist/dist))
                if not petal.color.proper():
                    petal.colorize(petal.og_color*(255/petal.og_color.max()))
                
    def cpy(self):
        flowers_copy = []
        vase_copy = Vase([]) # obviously not
        for flower in self.flowers:
            flowers_copy.append(flower.cpy())
        self_copy = Bouquet(flowers_copy, vase_copy)
        return self_copy

# canvas class
class Canvas:
    '''The canvas is a 2D rendering of the 3D bouquet,
to be output to the terminal. (x,y,z) maps to canvas.canvas
[ z+HEIGHT//2 ][ x+WIDTH//2 ] ~ish.'''
    def __init__(self, width, height):
        self.w = width
        self.h = height
        self.canvas = [[('  ', vec(0,0,0)) for i in range(width)]
                       for j in range(height)]
        self.bouquet = None
    
    def draw_bouquet(self, bouquet, vantage):
        self.canvas = [[('  ', vec(0,0,0)) for i in range(self.w)]
                       for j in range(self.h)]
        self.bouquet = bouquet
        notes = bouquet.vase.edges[:]
        for flower in bouquet.flowers:
            notes += flower.petals
        notes = [((vantage*(-1) + note.pos).l, note) for note in notes]
        notes.sort(reverse = True)
        
        norm = vantage*(RADIUS / (vantage.l**0.5))
        # plane is norm.x * x + norm.y * y + norm.z * z = norm.l
        a = vec(0,0,1).cross(norm) # horizontal basis vector, unscaled
        b = norm.cross(a) # vertical basis vector, unscaled
        # scaling
        a = a*(1/a.l**0.5)
        b = b*(1/b.l**0.5)
        det = a.x*b.y - a.y*b.x
        # we can safely assume det is nonzero!!
        # inverse matrix:
        inv1 = vec(b.y, -b.x, 0)*(1/det)
        inv2 = vec(-a.y, a.x, 0)*(1/det)
        
        grid = [[None for i in range(RADIUS*2)] for j in range(RADIUS*2)]
        for d, note in notes:
            # find plane-line intersection x
            t = (norm.l - note.pos.dot(norm))/(vantage.dot(norm) - note.pos.dot(norm))
            x = vantage*t + note.pos*(1-t)
            # translate into point (p,q) on the a-b plane
            p = inv1.dot(x-norm)
            q = inv2.dot(x-norm)
            # discretize
            if (p*p + q*q) <= RADIUS*RADIUS:
                p,q = int(p), int(q)
                if not (((10+len(bouquet.vase.radii))>note.pos.z >= -12) and ((note.pos.x**2 + note.pos.y**2) < (bouquet.vase.radii[int(note.pos.z)-10]**2))):
                    grid[q][p] = note
        
        # take screen-slot, centered at (0,0)
        # so p in [-WIDTH//2, WIDTH//2] and q in [-HEIGHT//2, HEIGHT//2]
        for p in range(-WIDTH//2, WIDTH//2):
            for q in range(-HEIGHT//2, HEIGHT//2):
                note = grid[q][p]
                if note is not None:
                    display_char = char_perception(note, vantage)
                    canvas.canvas[q+HEIGHT//2][p+WIDTH//2] = (display_char, grid[q][p].color)
    
    def display(self, message):
        print("\u001b[?25l")
        #blit canvas to screen
        print(f"\u001b[{self.h+1}A", end = "") # move cursor to top
        for y in range(self.h):
            print("\u001b[K", end = f"") # clear line
            for x in range(self.w):
                char, color = self.canvas[y][x]
                if TRUECOLOR:
                    print(f"\u001b[38;2;{round(color.x)};{round(color.y)};{round(color.z)}m{char}" if char != '  ' else '  ', end="")
                else:
                    print(f"\u001b[38;5;{color.color_256()}m{char}" if char != '  ' else '  ', end = "")
            print('') # next line
        
        #wait
        print(message, end = "")
        time.sleep(WAIT_TIME)

# templates
TEMPLATES = []

lavender = Flower()
lavender.pos = vec(1,0,0)
root = Petal(vec(0,0,0), '-')
lavender.petals.append(root)
for d in range(1, 11):
    leaf = Petal(vec(d,0,0), '-')
    theta = rand(3*math.pi/4, 5*math.pi/4)
    phi = rand(-math.pi/4, math.pi/4)
    leaf.pos += vec(1.5,0,0).rot(theta, phi)
    lavender.petals.append(leaf)
for d in range(1, 11):
    leaf = Petal(vec(d,0,0), '-')
    theta = rand(3*math.pi/4, 5*math.pi/4)
    phi = rand(-math.pi/4, math.pi/4)
    leaf.pos += vec(1.5,0,0).rot(theta, phi)
    lavender.petals.append(leaf)
for d in range(1, 11):
    leaf = Petal(vec(d,0,0), '-')
    theta = rand(3*math.pi/4, 5*math.pi/4)
    phi = rand(-math.pi/4, math.pi/4)
    leaf.pos += vec(1.5,0,0).rot(theta, phi)
    lavender.petals.append(leaf)
TEMPLATES.append(lavender)

bluebell = Flower()
bluebell.pos = vec(1,0,0)
for d in range(5, 30, 4):
    bell = Petal(vec(d/4,0,0), '^')
    phi = math.pi/2-0.001
    bell.pos += vec(0,0,3.5)
    bluebell.petals.append(bell)

TEMPLATES.append(bluebell)

bobble = Flower()
bobble.pos = vec(1,0,0)
for x in range(-2, 4):
    for y in range(-2, 3):
        for z in range(-2, 3):
            if x**2 + y**2 + z**2 <= 25:
                petal = Petal(vec(x,y,z), 'o')
                bobble.petals.append(petal)
TEMPLATES.append(bobble)
TEMPLATES.append(bobble)

daisy = Flower()
daisy.pos = vec(1,0,0)
daisy.petals = [Petal(vec(1,0,0), 'o')]
for i,j,k in [(-1, 0, 0), (-1, -1, math.pi/4), (-1, 1, math.pi*7/4), (0, 1, math.pi/2), (0,-1,math.pi/2), (1, 1, math.pi/4), (1, 0, 0), (1, -1, math.pi*7/4)]:
    daisy.petals.append(Petal(daisy.pos + vec(i,j,0), '-'))
    daisy.petals[-1].theta = k
    
TEMPLATES.append(daisy)

#rose = Flower() ? by any other name idk
        

vase = Vase()

flowers = [Flower().new_flower(None) for i in range(NUM_FLOWERS)]
canvas = Canvas(WIDTH, HEIGHT)
bouquet = Bouquet(flowers, vase)

y = input('''
Select a colour palette, or enter your own:
 [1] monochrome
 [2] complementary
 [3] analogous
 [4] triadic
 [5] rainbow!
 [6] pick your own :)

 :: ''')
while y not in ['1','2','3','4','5','6']:
    y = input("One more time, please :) :: ")

if y == '1':
    bouquet.colorize("monochromatic")
elif y == '2':
    bouquet.colorize("complementary")
elif y == '3':
    bouquet.colorize("analogous")
elif y == '4':
    bouquet.colorize("triadic")
elif y == '5':
    bouquet.colorize("rainbow!")
else:
    color_scheme = []
    y = input('''
All right, just enter rgb values until you're done, then enter
anything else! Use the format r,g,b (comma-separated, no spaces).

 :: ''')
    while True:
        k = y.split(',')
        if len(k) != 3:
            break
        if k[0] not in [str(i) for i in range(256)]:
            break
        if k[1] not in [str(i) for i in range(256)]:
            break
        if k[2] not in [str(i) for i in range(256)]:
            break
        r,g,b = [int(i) for i in k]
        color_scheme.append((r,g,b))
        y = input("\n :: ")
    bouquet.colorize(color_scheme)

for i in range(6):
    flowers += [Flower().new_flower(bobble)]
    flowers[-1].pos = vec(rand(-5, 5), rand(-5, 5), rand(-5, 5))
    for petal in flowers[-1].petals:
        petal.pos += flowers[-1].pos
        petal.set_og_color(vec(255,0,240))
        petal.colorize(vec(255,0,240))

vantage = vec(150, 1, -70)

print('''Great stuff! Now, we'll run a nuclear-physics-based
simulation to pull the bouquet together. Make sure you're
in full-screen mode, and sit tight!''')

time.sleep(3)

for i in range(REID_STEPS):
    bouquet.liquid_drop_step()
    bouquet.shade(vantage)
    canvas.draw_bouquet(bouquet, vantage)
    canvas.display('['+'#'*int(40*i/REID_STEPS)+' '*(40-int(40*i/REID_STEPS))+']')

# finally, re-center:
totale = vec(0,0,0)
total_n = 0
for flower in bouquet.flowers:
    for petal in flower.petals:
        totale += petal.pos
        total_n += 1
totale = totale * (-1/total_n)
for flower in bouquet.flowers:
    flower.pos += totale
    for petal in flower.petals:
        petal.pos += totale

time.sleep(4)
while True:
    bouquet.rotate(THETA, PHI)
    bouquet.shade(vantage)
    canvas.draw_bouquet(bouquet, vantage)
    canvas.display("HAPPY BIRTHDAY BEE JACKSON :) I LOVE YOU BOSS <3")

