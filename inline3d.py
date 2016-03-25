# modifié pour du python 3

# importation de diverses bibliotheques
from copy import copy
from random import choice
import math
import random
import string

# classe Coord3D
# correspond à une position dans un espace vectoriel sur {-1;0;1}^3 muni des loi interne __add__ et de la loi externe __mul__
class Coord3D:
    def __init__(self,a,b,c):
        self.x = a
        self.y = b
        self.z = c
    def __mul__(self,k):
        resultat = copy(self)
        resultat.x *= k
        resultat.y *= k
        resultat.z *= k
        return resultat
    def __add__(self,u):
        resultat = copy(self)
        resultat.x += u.x 
        resultat.y += u.y
        resultat.z += u.z
        return resultat
    def __eq__(self,u):
        if self.__hash__() == u.__hash__():return True
        else: return False
    def __hash__(self): return self.__str__().__hash__()
    def linear(self,dim):return self.x+self.y*(dim-1)+self.z*(dim-1)*dim
    def __str__(self): return '('+', '.join([str(self.x),str(self.y),str(self.z)])+')'
    def __unicode__(self): return '' + self.__str__()

# classe Joueur: 
# ia en cours d'implementation
class Joueur:
  def __init__(self,n,t='',user='',passwd='',ia={}):
        self.nom = n
        self.texte = t
        self.ia = ia
        self.user=user
        self.passwd=passwd
        self.nbParties=0
        self.nbVictoires=0
        self.nbDefaites=0
        self.nbMatchNuls=0
  def __str__(self): return '- '.join([self.nom,self.texte])
  def __unicode__(self): return '' + self.__str__()

# classe Piece:
# 2 références, un joueur et une position
# Id pourra correspondre à l'ordre dans lequel la pièce est jouée (pas encore définitif sur ce point)
class Piece:
  def __init__(self,u,v,index):
        self.joueur = u
        self.position = v
        self.ID = index
  def __hash__(self): return self.position.__hash__()
  def __str__(self): return '::'.join([str(self.ID),self.joueur.__str__(),self.position.__str__()])
  def __unicode__(self): return ''+self.__str__()
  def __copy__(self): return Piece(self.joueur,copy(self.position),self.ID)

# tentative d'implementation d'un réseau de neurones à propagation rétroactive

# see  bpnn.py
# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>
def rand(a, b):
    return (b-a)*random.random() + a
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m
def sigmoid(x):
    return math.tanh(x)
def dsigmoid(y):
    return 1.0 - y**2
class NN:
    def __init__(self, ni, nh, no):
        self.ni = ni + 1
        self.nh = nh
        self.no = no
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)    
    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError #, 'wrong number of inputs'
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)    
        return self.ao[:]
    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError #, 'wrong number of target values'
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error
    def test(self, patterns):
        for p in patterns:
            print (p[0], '->', self.update(p[0]))
    def weights(self):
        print ('Input weights:')
        for i in range(self.ni):
            print (self.wi[i])
        print
        print ('Output weights:')
        for j in range(self.nh):
            print (self.wo[j])    
    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        for i in xrange(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print ('error %-14f' % error)

# fin des élucubrations
                

# classe principale
class Classic:
    def __init__(self,j1,j2,d=3,nb=3):
# par defaut, morpion 3D de dimension 3x3x3
        if d <3: self.dim = 3
        else: self.dim = d
        if nb < 3: self.nbAlign = 3
        else :
            if nb > self.dim : self.nbAlign = self.dim
            else: self.nbAlign = nb

        self.joueur1 = j1
        self.joueur2 = j2
# maj des stats de chaque joueur
        j1.nbParties += 1
        j2.nbParties += 1
# initialisation des tableaux, des dico et des listes
        self.tableau = {}
        self.pattern1 = {}
        self.pattern2 = {}
        self.trouve = []
        self.end = False
        self.vainqueur = None
        self.compteur = 0
        self.typ = 'classic'
        self.joueurActif = choice([j1,j2])

# retourne une copie de la liste passé en paramètre ou bien le tableau du jeu, dont les pieces sont des copies aussi mais pas les joueurs        
    def getPieces(self,pieces=None):
        if pieces==None: pieces = self.tableau.values()
        tmp = []
        for i in pieces:tmp.append(i.__copy__())
        return tmp
#vérifie que les coordonnées sont bien des entiers positifs inférieur à SELF.DIM
    def valid(self,u):
        if (u.position.x < 0) or (u.position.x > self.dim-1) or (u.position.x != int(u.position.x)): raise self.HorsLimite(pieces)
        elif (u.position.y < 0) or (u.position.y > self.dim-1) or (u.position.y != int(u.position.y)): raise self.HorsLimite(pieces)
        elif (u.position.z < 0) or (u.position.z > self.dim-1) or (u.position.z != int(u.position.z)): raise self.HorsLimite(pieces)
        return True
# change le joueur actif
    def suivant(self):
        if self.joueurActif == self.joueur1: self.joueurActif = self.joueur2
        else : self.joueurActif = self.joueur1
# cherche si il y a NBALIGN pieces de la liste TAB alignés sur la droite V, passant par la position O
    def cherche(self,o,v,tab):
        solution = []
        idx = 1
        compt = 1
        avant = True
        apres = True
        
#on compte les pieces de proche en proche autour de la piece O, toujours sur le vecteur V        
        while (idx < self.nbAlign) and ( avant or apres ):
            if (o.position + (v*(-idx))).__hash__() in tab:
# verifie la prochaine piece devant la piece u                
                if avant:
                    tmp =(o.position + (v*(-idx))).__hash__()
                    if tab[tmp].joueur == o.joueur: 
                        compt += 1
                        solution.append(tab[tmp])
                    else: avant = False
            else: avant = False
            if (o.position + (v*(idx))).__hash__() in tab:
# verifie la prochaine piece apres u
                if apres:
                    tmp= (o.position + (v*(idx))).__hash__()
                    if tab[tmp].joueur == o.joueur: 
                        compt += 1
                        solution.append(tab[tmp])
                    else: apres = False
            else: apres = False
            idx += 1
# compt vaut le nombre de piece avant u + le nombre de piece apres u de la meme couleur que u
        if compt >= self.nbAlign: 
            solution.append(o)
            self.trouve.append( solution )
            return True
        else:
            return False        
# sur un rubick' Cube, il y a 26 petits cubes autour d'un cube central inacessible nommé machin
# il y a donc 13 vecteurs linéairement indépendants 2 à 2 ET qui passe par machin
# on recherche des pions alignés sur chacun de ces vecteurs
    def check(self,u,piece=None):
        if piece == None:
            piece = self.tableau
        if (    self.cherche(u,Coord3D(1,0,0),piece) 
            or self.cherche(u,Coord3D(0,1,0),piece) 
            or self.cherche(u,Coord3D(0,0,1),piece) 
            or self.cherche(u,Coord3D(1,1,0),piece) 
            or self.cherche(u,Coord3D(1,0,1),piece) 
            or self.cherche(u,Coord3D(0,1,1),piece) 
            or self.cherche(u,Coord3D(-1,1,0),piece) 
            or self.cherche(u,Coord3D(0,-1,1),piece) 
            or self.cherche(u,Coord3D(1,0,-1),piece) 
            or self.cherche(u,Coord3D(1,1,-1),piece) 
            or self.cherche(u,Coord3D(1,-1,-1),piece) 
            or self.cherche(u,Coord3D(-1,1,-1),piece) 
            or self.cherche(u,Coord3D(1,1,1),piece)
        ):
# on met à jour la partie, mais aussi les stats des joueurs en cas de victoire de l'un d'eux
            if not self.end:
                self.end = True
                self.vainqueur = u.joueur
            else:
                if self.vainqueur != u.joueur: self.vainqueur = None

    def updateStat(self):
        if self.end:
            if self.vainqueur != None:
                self.vainqueur.nbVictoires += 1
                if self.vainqueur == self.joueurActif:self.suivant()
                self.joueurActif.nbDefaites +=  1
            else:
                j1.nbMatchNuls += 1
                j2.nbMatchNuls += 1

# retourne un couple entree/sortie pour le réseau de neurones
# couple de tableau représentant la situation à l'instant présent
    def historise(self,i,o):
        if o.joueur == self.joueur1:self.pattern1[(''.join(str(self.mkInput(i,[self.joueur1,self.joueur2])))).__hash__()]=(i,o)
#        elif o.joueur == self.joueur2: self.pattern2[(u''.join(str(self.mkInput(i,[self.joueur2,self.joueur1])))).__hash__()]=(i,o)
        else: self.pattern2[(''.join(str(self.mkInput(i,[self.joueur2,self.joueur1])))).__hash__()]=(i,o)

# mise à jour du reseau de neurones: a faire
    def memorise(self):
        pass

# ajoute un pion: exception si la case est déjà occupée
    def ajoute(self,p):
        i=self.getPieces()
        u=Piece(self.joueurActif,Coord3D(p.position.x,p.position.y,p.position.z),self.compteur)
        if self.valid(u):
            if u.__hash__() in self.tableau: raise self.ExisteDeja(p)
            u.ID = self.compteur
            self.tableau[u.position.__hash__()] = u
            self.historise(i,u)
            self.compteur += 1
            self.check(u)
            if self.end:
                self.memorise()
            else:self.suivant()
        else: raise e        
# retourne une copie de liste de pions, en gardant la notion du joueur pour les pieces contenues dans la liste garde, et en enlevant la notion du joueur pour les pieces contenues dans la liste enleve
    def mkIndifferent(self,garde=[],enleve=[]):
        resultat= []
        for i in self.tableau.values():
            tmp = copy(i)
            if i not in enleve:
                if not i in garde:
                    tmp.joueur = None
                resultat.append(tmp)
        return resultat                               
    
# retourne un tableau binaire double à partir d'un tableau de pieces pour l'entrée du réseau de neurones
    def mkInput(self,pieces,q):
        resultat = []        
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    resultat.append(0)
                    resultat.append(0)
        for p in pieces:
            if p.joueur == q[0]: resultat[int((p.position).linear(self.dim)*2)]=1
            elif p.joueur == q[1]: resultat[int((p.position).linear(self.dim)*2+1)]=1
            else:
                resultat[int((p.position).linear(self.dim)*2)]=1
                resultat[int((p.position).linear(self.dim)*2+1)]=1
        return resultat
    
# retourne un tableau binaire simple à partir des pieces gagnantes ou bien pieces remarquables pour la sortie du réseau de neurones
    def mkOutput(self,pieces):
        resultat=[]
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim): resultat.append(0)
        for i in pieces: resultat[int((i.position).linear(self.dim))]=1
        return resultat

# retourne une copie de la liste de pieces mais on echange les joueurs
    def inversion(pieces,j):
        tmp = self.getPieces(pieces)
        for i in tmp:
            if i.joueur == j[0]: i.joueur = j[1]
            else: i.joueur = j[0]
        return tmp

# retourne une liste de pions apres avoir appliqué une translation de vecteur V
    def translation(self,pieces,v): 
        tmp = self.__getPieces(pieces)
        for i in tmp: i.position += v 
        return tmp

# retourne une liste de pions apres avoir appliqué une symétrie planaire de vecteur directeur Ox 
    def symx(self,pieces):
        tmp = self.translation(pieces,Coord3D(-self.dim/2.0,0,0))
        for i in tmp: i.position.x = - i.position.x + self.dim/2.0 -1
        return tmp

# retourne une liste de pions apres avoir appliqué une symétrie planaire de vecteur directeur Oy    
    def symy(self,pieces):
        tmp = self.translation(pieces,Coord3D(0,-self.dim/2.0,0))
        for i in tmp: i.position.y = - i.position.y + self.dim/2.0 -1
        return tmp

# retourne une liste de pions apres avoir appliqué une symétrie planaire de vecteur directeur Oz
    def symz(self,pieces):
        tmp = self.translation(pieces,Coord3D(0,0,-self.dim/2.0))
        for i in tmp: i.position.z = - i.position.z + self.dim/2.0 -1
        return tmp

# retourne une liste de pions apres avoir appliqué une rotation nPI Pi/2.0 autour de l'axe Z     
    def rotz(self,pieces,nPI=1):
        nb = nPI % 4
        if nb !=0:
            for k in range(nb):            
                tmp2 = self.translation(pieces,Coord3D(-self.dim/2.0,-self.dim/2.0,0))
                for i in tmp2:
                    tmp = copy(i.position)
                    i.position.y = tmp.x + self.dim/2.0
                    i.position.x = - tmp.y + self.dim/2.0 -1
        else:
            tmp2 = self.getPieces(pieces)
        return tmp2

# retourne une liste de pions apres avoir appliqué une rotation nPI Pi/2.0 autour de l'axe Y    
    def roty(self,pieces,nPI=1):
        nb = nPI % 4
        if nb != 0:
            for k in range(nb):
                tmp2 = self.translation(pieces,Coord3D(-self.dim/2.0,0,-self.dim/2.0))
                for i in tmp2:
                    tmp = copy(i.position)
                    i.position.x = tmp.z + self.dim/2.0 
                    i.position.z = - tmp.x + self.dim/2.0 - 1
        else:
            tmp2 = self.getPieces(pieces)
        return tmp2

# retourne une liste de pions apres avoir appliqué une rotation nPI Pi/2.0 autour de l'axe X    
    def rotx(self,pieces,nPI=1):
        nb = nPI % 4
        if nb!=0:
            for k in range(nb):
                tmp2 = self_translation(pieces,Coord3D(0,-self.dim/2.0,-self.dim/2.0))
                for i in tmp2:
                    tmp = copy(i.position)
                    i.position.y = - tmp.z + self.dim/2.0 -1
                    i.position.z = tmp.y + self.dim/2.0  
        else:
            tmp2 = self.getPieces(pieces)
        return tmp2

# retourne un dictionnaire de couple entrée/sortie de tableau destiné au réseau de neurones    
# interet = liste de pions remarquables, la plupart du temps, les pions gagnants
# j0 tableau de joueur [j1,j2] ou bien [j2,j1], à préciser si mkpattern traite une liste de pions n'appartenant pas à la partie en cours
    def mkPattern(self,interet,j0=None):    
        liste = {}        
        pieces1 = self.mkIndifferent(interet)
        pieces2 = self.getPieces()
        if j0 == self.joueur2:
            j=[self.joueur2,self.joueur1]
        else:
            j=[self.joueur1,self.joueur2]
        u = self.getPieces(interet)
# on va opérer à toutes les compositions de rotations, symetries et inversions et créer autant de résultats qu'il y a de combinaisons distinctes
        if self.typ != 'Gravitic':
            for a in ( lambda x: x, lambda x: self.rotx(x),lambda x: self.rotx(x,2),lambda x:self.rotx(x,3)):
                for b in ( lambda y: y, lambda y: self.roty(y),lambda y: self.roty(y,2),lambda y:self.roty(y,3)):
                    for c in ( lambda z: z, lambda z: self.rotz(z),lambda z: self.rotz(z,2),lambda z:self.rotz(z,3)):
                        for d in (lambda s: s, lambda s:self.symx(s)):
                            tmp2 = map(a,map(b,map(c,map(d,[pieces1,pieces2,u]))))
                            tmp = [self.mkInput(tmp2[0],j),self.mkInput(tmp2[1],j),self.mkOutput(tmp2[2])]
                            liste[(':'.join([self.typ,str(self.dim),str(self.nbAlign)])+''.join(str(tmp[0]))).__hash__()]=(tmp[0],tmp[2])
                            liste[(':'.join([self.typ,str(self.dim),str(self.nbAlign)])+''.join(str(tmp[1]))).__hash__()]=(tmp[1],tmp[2])           
        else:
          for c in ( lambda z: z, lambda z :self.rotz(z),lambda z: self.rotz(z,2),lambda z:self.rotz(z,3)):
            for d in (lambda s: s, lambda s: self.symx(s)):
              tmp2 = map(c,map(d,[pieces1,pieces2,u]))
              tmp = [self.mkInput(tmp2[0],j),self.mkInput(tmp2[1],j),self.mkOutput(tmp2[2])]
              liste[('::'.join([self.typ,str(self.dim),str(self.nbAlign)])+''.join(str(tmp[0]))).__hash__()]=(tmp[0],tmp[2])
              liste[('::'.join([self.typ,str(self.dim),str(self.nbAlign)])+''.join(str(tmp[1]))).__hash__()]=(tmp[1],tmp[2])            
        return liste
    def mouvementsPossibles(self):
        resultat = {}
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    tmp = Coord3D(i,j,k)
                    if tmp.__hash__() not in self.tableau:
                        resultat.append((tmp))
        return resultat
    
    def askMove(self,piece,deep,j0=None):
        if j0 == None:
            j0=self.joueurActif
        if self.end or deep ==0:
            return None
        else:
            resultat = {}
            for i in range(self.dim):
                for j in range( self.dim):
                    for k in range(self.dim):
                        tmp = copy(self)
                        try:
                            tmp.ajoute(Piece(tmp.joueurActif,Coord3D(i,j,k),0))
                            if tmp.end:
                                resultat.append(tmp.lastMove)
                            else:
                                pass
                            
                        except:
                            pass
                        
# exception pour une partie "CLASSIQUE"    
    class ExisteDeja(Exception):
        def __init__(self,p):
            Exception.__init__(self)
            self.data = [p]
# exception pour une partie quelconque
    class HorsLimite(Exception):
        def __init__(self,p):
            Exception.__init__(self)
            self.data = [p]

class Gravitic(Classic):
    def __init__(self,j1,j2,d=4,nb=4): 
        Classic.__init__(self,j1,j2,d,nb)
        self.typ = 'Gravitic'

    def mouvementsPossibles(self):
        resultat = {}
        for i in range(self.dim):
            for j in range(self.dim):
                tmp = Coord3D(i,j,self.dim - 1)
                if tmp.__hash__() not in self.tableau:
                    resultat.append((tmp))
        return resultat

# les pions descendent le long de l'axe Z et s'accumulent en colonne
    def ajoute(self,p):
        i=self.getPieces()        
        u=Piece(self.joueurActif,Coord3D(p.position.x,p.position.y,p.position.z),self.compteur)
        if self.valid(u): 
            u.position.z = 0
            while (u.__hash__() in self.tableau) and (u.position.z <= self.dim - 1 ):u.position.z += 1
            if not (u.__hash__() in self.tableau) and (u.position.z <= self.dim - 1): 
                u.ID = self.compteur
                self.tableau[u.position.__hash__()] = u
                self.historise(i,u)
                self.compteur += 1
                self.check(u)
                if self.end:
                    self.memorise()
                else:self.suivant()
            else: raise self.ColonnePleine(p)
            
# exception lors d'une partie "GRAVITIC"
    class ColonnePleine(Exception):
        def __init__(self,p):
            Exception.__init__(self)
            self.data = [p]

class AntiGravitic(Classic):
    def __init__(self,j1,j2,d=5,nb=5):
        Classic.__init__(self,j1,j2,d,nb)
        self.typ = 'AntiGravitic'
        self.dernierMouvement = Piece(None,Coord3D(-1,-1,-1),-1)
        self.drop = []

# detection si la piece se trouve bien sur un bord du tableau        
    def __borderLine(self,u):
        resultat = []
        if u.position.x == 0: resultat.append(Coord3D(1,0,0))
        elif u.position.x == self.dim-1: resultat.append(Coord3D(-1,0,0))
        if u.position.y == 0: resultat.append(Coord3D(0,1,0))
        elif u.position.y == self.dim-1: resultat.append(Coord3D(0,-1,0))
        if u.position.z == 0: resultat.append(Coord3D(0,0,1))
        elif u.position.z == self.dim-1: resultat.append(Coord3D(0,0,-1))
        return resultat    

    def mouvementsPossibles(self):
        resultat = {}
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    tmp = Coord3D(i,j,k)
                    tmp2 =  __borderline(tmp)
                    if tmp2 != []:
                        for l in tmp2:
                            resultat.append((tmp,l))
        return resultat
    
# les pions sont poussés de l'exterieur vers l'intérieur et en cascade
# on ne peut pas ejecter le dernier pion posé par l'adversaire
    def ajoute(self,p,direction):
        i=self.getPieces()
#
        u=Piece(self.joueurActif,Coord3D(p.position.x,p.position.y,p.position.z),self.compteur)
#
        v=Coord3D(direction.x,direction.y,direction.z)
        tmp = None
        if not self.valid(u) or (v not in self.__borderLine(u)): raise self.HorsLimite(p) 
        j=0
        while (u.position + (v*j)).__hash__() in self.tableau: j = j+1        
        try:
            tmp2 = self.tableau[(u.position + (v*(j - 1))).__hash__()]
        except:
            tmp2 = None
        if (tmp2 == self.dernierMouvement) and (j == self.dim ): raise self.MouvementIllegal(p,direction)
        for k in range(j): 
            tmp = self.tableau[(u.position+(v*(j-k-1))).__hash__()]
            if (j-k) < self.dim : 
                tmp.position = u.position+(v*(j-k))
                self.tableau[tmp.position.__hash__()] = tmp
            else: 
                self.drop.append(self.tableau.pop((u.position+(v*(j-k-1))).__hash__()))
        self.tableau[u.position.__hash__()] = u
        self.dernierMouvement = u
        self.historise(i,u)
        self.compteur += 1
        for k in range(j): self.check(self.tableau[(u.position+(v*(j-k-1))).__hash__()])
        if self.end:
            self.memorise()
        else:self.suivant()
        
# exception lors d'une partie "ANTIGRAVITIC"
    class MouvementIllegal(Exception):
        def __init__(self,p,v):
            Exception.__init__(self)
            self.data = [p.__str__(),v.__str__()]
                
print ('version 0.4a')
if __name__ == "__main__":
    joueur1 = Joueur("Toto")
    joueur2 = Joueur("Taratata")
    partie = AntiGravitic(joueur1,joueur2)
    compteur = 0
    coupsDIR = [
             [Piece(None,Coord3D(2,0,0),0),Coord3D(0,1,0)],
             [Piece(None,Coord3D(0,0,0),0),Coord3D(0,1,0)],
             [Piece(None,Coord3D(2,0,0),0),Coord3D(0,1,0)],
             [Piece(None,Coord3D(0,0,0),0),Coord3D(0,1,0)],
             [Piece(None,Coord3D(2,0,0),0),Coord3D(0,1,0)],
             [Piece(None,Coord3D(0,0,0),0),Coord3D(0,1,0)],
             [Piece(None,Coord3D(2,0,0),0),Coord3D(0,1,0)],
             [Piece(None,Coord3D(0,4,0),0),Coord3D(0,-1,0)],
             [Piece(None,Coord3D(2,0,0),0),Coord3D(0,1,0)]
# a completer             
             ]


    for i in coupsDIR:
        if not(partie.end):
            compteur += 1
            try:
                partie.ajoute(i[0],i[1])
                print (partie.joueurActif.__str__(),compteur)
            except partie.MouvementIllegal as e:
                print (" ".join(["***","".join(e.data),"***"]))
            # meme joueur rejoue
            for u in partie.getPieces():print (u)
    partie.updateStat()
    if partie.end:
        if partie.vainqueur != (None):
            print ("Le joueur " + str(partie.vainqueur.nom) + " a gagné.")
            for i in partie.trouve:
                for j in i:
                    print(j)
        else:
            print ("Match nul.")
    else:
        print ("partie non terminée.")

else:
    print ("a finir: askmove + recursion")
    print ("a finir: memorize")
    print ("a commencer: recall")
