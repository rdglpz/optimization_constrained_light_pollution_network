import numpy as np
import itertools as it

class NetworkFitness():
    """
    
    """
    
    
    def __init__(self, 
                 NTLI, 
                 EAM, 
                 sensitivity,
                 local_variograms,
                 local_variograms_m,
                 coordinates,
                 network,
                 r,
                 th,
                 alpha = 0.5
                ):

        print("Selct cost functions: \n 'xor','max' or 'cover'")
        self.NTLI = NTLI
        self.EAM = EAM
        self.sensitivity = sensitivity
        self.local_variograms = local_variograms
        self.local_variograms_m = local_variograms_m
        self.coordinates = coordinates
        self.init_network = network
        x = int(len(network)/2)
        self.r = r
        self.th = th
        self.c = self.generateCombinations(x, r)
        self.alpha = alpha
        
    
    def plotLocations(self, X):
        
        locations = np.zeros(self.EAM.shape)
        
        for loc  in X.reshape(-1,2):
            locations[tuple(loc)] = 1
        
        return locations
        
    
    def J(self, qp):
        '''
        
        '''
        
        
        
        q = self.init_network
        r = self.r
        
        #recovering the number of sensors
        n = int(len(q)/2)
        
        maximum_average = np.average(np.sort(self.NTLI.flatten())[-n:])
        
        #$C^n_r = n!/(r!(n-r)!)$ Eq (2)
        K = list(it.combinations(np.arange(n), r))
        
        maximum = 0
        
        qc = np.copy(q)
        
        for j, k in enumerate(K):
        
            #print("Combination No:", j)
        
            #new copy of the network for every new combination
            qc = np.copy(q)

            #index of selected sensors
            for c in k:
         
                s_ix = [c*2, c*2 + 1]
            
                #crossover, we substitute the sensors of the selected 
                #index for the alternative positions
                
                qc[s_ix] = qp[s_ix]
                
        
            #covers
            #positions = qc.reshape(-1, 2)
            
            #covers = self.getCapturedLightPollutionCovers(positions)
            
            #LP = np.sum(np.max(covers, axis=0))
            #PLP = np.sum(self.NTLI*self.EAM)
            #F1 = (LP/PLP)
            
            F1 = self.f1(qc)
            
           
            
            #F2 = self.getConcentrationLightPollution(positions)
            #F2 = self.getPuntualLightPollution(qc)
            f3 = self.f3(qc)
            
            f = self.alpha*F1 + (1 - self.alpha)*f3
            
   
            if f > maximum:
    
                maximum = f
        
                self.bestComb = np.copy(qc)
            

        return -maximum
    
    def f1(self, X):
        
        positions = X.reshape(-1, 2)
        covers = self.getCapturedLightPollutionCovers(positions)
        LP = np.sum(np.max(covers, axis=0))
        PLP = np.sum(self.NTLI*self.EAM)
        return LP/PLP
        
        
    
    def getCapturedLightPollutionCovers(self, positions):
        """
        positions: n coordinate pairs [[x1,y1],[x2,y2], [xn,yn]]
        """
        
        FDNTLI = self.NTLI
        
        EAM = self.EAM
        
        covers = np.zeros((len(positions)+1, FDNTLI.shape[0], FDNTLI.shape[1]))
        
        for i, p in enumerate(positions):
            
            ub = np.max(FDNTLI)**2/2
            ix = self.validate_coordinates(p[0], p[1])
            
            if len(ix)>0:
                
                tvar = self.local_variograms[ix][0]
                tvar_m = self.local_variograms_m[ix][0]
                outofrange = (tvar_m==0)*ub
                inrange = (tvar_m==1)*tvar
                M = inrange + outofrange
                map0to1 = ((ub - M)/(ub))
                map0to1 = map0to1*(map0to1 > self.th)
                #map0to1 = map0to1*(map0to1)
                
                #map021 is ok
                #print("max", np.max(map0to1))
                #print("min", np.min(map0to1))
                #covers[i+1] = np.copy(tvar*tvar_m*EAM*FDNTLI)
                covers[i+1] = map0to1*FDNTLI*tvar_m*EAM
                #print(np.max(covers[i+1]))
                
                
        return covers
    
    def getCapturedLightPollutionInCriticalPoints(self, positions):        
            
        """
        positions: n coordinate pairs [[x1,y1],[x2,y2], [xn,yn]]
        """
       
        FDNTLI = self.NTLI
        
        EAM = self.EAM
        
        covers = np.zeros((len(positions)+1, FDNTLI.shape[0], FDNTLI.shape[1]))
        
        
        FDNTLI_Flatten = FDNTLI.flatten()
        
        n = len(positions)
        
        th = np.sort(FDNTLI_Flatten)[::-1][n-1]
        
        MH = FDNTLI>=th
        
        for i, p in enumerate(positions):
            
            ub = np.max(FDNTLI)**2/2
            ix = self.validate_coordinates(p[0], p[1])
            
            if len(ix)>0:
                
                tvar = self.local_variograms[ix][0]
                tvar_m = self.local_variograms_m[ix][0]
                outofrange = (tvar_m==0)*ub
                inrange = (tvar_m==1)*tvar
                M = inrange + outofrange
                map0to1 = ((ub - M)/(ub))
                map0to1 = map0to1*(map0to1 > self.th)
                #map0to1 = map0to1*(map0to1)
                
                #map021 is ok
                #print("max", np.max(map0to1))
                #print("min", np.min(map0to1))
                #covers[i+1] = np.copy(tvar*tvar_m*EAM*FDNTLI)
                covers[i+1] = map0to1*FDNTLI*tvar_m*EAM*MH
                #print(np.max(covers[i+1]))
                
                
        return covers
    
    def f3(self, X):
        
        FDNTLI = self.NTLI
        
        FDNTLI_Flatten = FDNTLI.flatten()
        
        X = X.reshape(-1, 2)
        
        n = len(X)
        
        th = np.sort(FDNTLI_Flatten)[::-1][n-1]
        
        MH = FDNTLI>=th
        
        C = self.getCapturedLightPollutionInCriticalPoints(X)
        
        
        maxC = np.max(C, axis = 0)
        maxC = np.sum(maxC)
        
        return maxC/np.sum(MH*FDNTLI)
        
        
        
    
    def getConcentrationLightPollution(self, positions):
        
        n = len(positions)
        
        n_avg = 0
        
        covers = self.getCapturedLightPollutionCovers(positions)
          
        for c in range(1, n + 1):
    
            #mascara del terrotorio que abarca el sensor c
            one_cover_mask = np.argmax(covers, axis = 0) == c


            #print(np.sum(one_cover_mask))

            if np.sum(one_cover_mask) == 0:

                avg_lp = 0

            else:

                one_cover_values = np.max(covers, axis = 0)
                sum_values = np.sum(one_cover_values*one_cover_mask)
                avg_lp = sum_values/np.sum(one_cover_mask)

            #print(avg_lp)
            n_avg += avg_lp
            
            
            
        return n_avg
    
    
    def getPuntualLightPollution(self, X):
        
        FDNTLI = self.NTLI
        
        P_bestcomb_bidim = X.reshape(-1, 2)

        n = len(P_bestcomb_bidim)

        Z = np.zeros((n + 1, FDNTLI.shape[0], FDNTLI.shape[1]))

        for i, ns in enumerate(P_bestcomb_bidim):
            
            t = tuple(ns)
            
            Z[i + 1][t] = FDNTLI[t]


        p = np.sum(np.max(Z, axis = 0))/n
        
        highest_pixels = np.average(np.sort(FDNTLI.flatten())[-n:])
        
        return p/highest_pixels

            
            
            
        
        
        
        
        
        
   

    def setCombinations(self, c):
        
        #combinations
        self.c = c
        
    def selectFitnessFunction(self,s):
        
        if s=="xor":
            self.f = self.xor
        elif s=="max":
            self.f = self.maximum
        elif s=="min":
            self.f = self.minimum
        elif s=="cover":
            self.f = self.bruteCoverage
        elif s=="explicability":
            self.f = self.explicability

        
    def validate_coordinates(self,iy,ix):
        """
        
        """
        
        iy = np.where(self.coordinates[:,0]==iy)
        ix = np.where(self.coordinates[:,1]==ix)
        
        return np.intersect1d(ix,iy)
        
        
   
  
    
    def coverage2(self, X):
        """
        
        """
        
        #get the numner of sensors.
        n_sensors = int(len(X)/2)
        
        
        
        #list of n elements with 2 
        #sensor_list = X.reshape(n_sensors,len(self.NLTI.shape))
        sensor_list = X.reshape(n_sensors, 2)

        #coverate_layers
        coverage = np.zeros((n_sensors+1, 
                             self.NTLI.shape[0], 
                             self.NTLI.shape[1]))
        
        for i, s in enumerate(sensor_list):

            sy, sx = s[0], s[1]
            ix = self.validate_coordinates(sy, sx)
            coverage[i+1] = np.zeros(self.NTLI.shape)
            
            if len(ix)>0:
                
                coordinates = self.coordinates[ix][0]
                # este esta de mas
                #pi = self.NTLI[coordinates[0]][coordinates[1]]
                
                
                #aqui tenemos que acotar al area de estudio, 
                #hacer esto en el primer paso de construir los variogramas 
                #y borrar esta operacion
                tvar = self.local_variograms[ix][0]*self.EAM
                tvar_m = self.local_variograms_m[ix][0]*self.EAM
                
                #outofrange = (tvar==0)*(pi**2/2)
                outofrange = (tvar_m==0)*(np.max(self.NTLI)**2/2)
                inrange = (tvar_m==1)*tvar

                
                #M = tvar+outofrange
                M = inrange+outofrange
                
                lb = (np.max(self.NTLI))**2/2
                map0to1 = (lb-M)/(lb)

                coverage[i+1] = map0to1*self.sensitivity
        return coverage
    
    def coverage_w_init_positions(self, X):
        """
        
        """
        
        #get the numner of sensors.
        n_sensors = int(len(X)/2)
        
        #select combination
        #s = X
        
        
    
        
        #list of n elements with 2 
        sensor_list = X.reshape(n_sensors,len(self.NTLI.shape))

        #coverate_layers
        coverage = np.zeros((n_sensors,self.NTLI.shape[0],self.NTLI.shape[1]))
        
        for i,s in enumerate(sensor_list):

            sy,sx = s[0],s[1]
            ix = self.validate_coordinates(sy,sx)
            coverage[i] = np.zeros(self.NTLI.shape)
            
            if len(ix)>0:
                
                coordinates = self.coordinates[ix][0]
                pi = self.NTLI[coordinates[0]][coordinates[1]]
                
                tvar = self.local_variograms[ix][0]
                tvar_m = self.local_variograms_m[ix][0]
                
                #outofrange = (tvar==0)*(pi**2/2)
                outofrange = (tvar_m==0)*(np.max(self.NTLI)**2/2)
                inrange = (tvar_m==1)*tvar

                
                #M = tvar+outofrange
                M = inrange+outofrange
                
                lb = (np.max(self.NTLI))**2/2
                map0to1 = (lb-M)/(lb)

                coverage[i] = map0to1*self.sensitivity
        return coverage
    
    def coverage_perturbing_sensors(self, X):
        
        
        
        #get the number of sensors.
        n_sensors = int((len(X)-1)/2)
        
        
        
        #list of n elements with 2 
        dim = len(self.NTLI.shape)
        #sensor_list = X.reshape(n_sensors, dim)
        
        
        
        
        #select combination
        ix_rcs = int(X[-1])
        
       
        
        Z = self.c[ix_rcs]
              
            
        #    self.init_network has the flattened version of the x,y sensor positions
        
        
        #vect_selection = np.zeros(int(len(self.init_network)/2))
        
        vect_selection = np.zeros(n_sensors)
        
        vect_selection[np.array(Z)] = 1
        
        #print("perturbin location of sensor:", vect_selection)
        
        sensor_list = self.init_network.flatten()
        
        for ix, x in enumerate(vect_selection):
    
            if x == 1:
                sensor_list[ix*2:ix*2+2] = X[ix*2:ix*2+2]
                
                
        #print(sensor_list)
        self.perturbed_sensor_list = sensor_list
        #we init the representation layers with zeros in a 3d matrix
        coverage = np.zeros((n_sensors, self.NTLI.shape[0],self.NTLI.shape[1]))
        
        #print(sensor_list.reshape(n_sensors, 2))
        for i, s in enumerate(sensor_list.reshape(n_sensors, 2)):
            
          # print(s)

            sy, sx = s[0], s[1]
            ix = self.validate_coordinates(sy, sx)
            #coverage[i] = np.zeros(self.NTLI.shape)
            
            if len(ix)>0:
                
                coordinates = self.coordinates[ix][0]
                #pi = self.NTLI[coordinates[0]][coordinates[1]]
                tvar = self.local_variograms[ix][0]
                
                tvar_m = self.local_variograms_m[ix][0]
                
                #outofrange = (tvar==0)*(pi**2/2)
                outofrange = (tvar_m==0)*(np.max(self.NTLI)**2/2)
                inrange = (tvar_m==1)*tvar

                
                #M = tvar+outofrange
                M = inrange+outofrange
                
                lb = (np.max(self.NTLI))**2/2
                map0to1 = (lb-M)/(lb)
                map0to1 = map0to1*(map0to1 > 0.95)

                coverage[i] = map0to1*self.sensitivity
                
        return coverage
        
        
        
        
        
    
    def coverMaps(self,X):
        """
        
        """
        n_sensors = int(len(X)/2)
        sensor_list = X.reshape(n_sensors,len(self.NTLI.shape))

        coverage = np.zeros((n_sensors,self.NTLI.shape[0],self.NTLI.shape[1]))
        
        for i,s in enumerate(sensor_list):

            sy,sx = s[0],s[1]
            ix = self.validate_coordinates(sy,sx)
            coverage[i] = np.zeros(self.NTLI.shape)
            
            if len(ix)>0:
                coverage[i] = self.local_variograms[ix][0]>0
            
        
        
        return coverage
                
    def maximum(self, X):
        """
        
        
        """
    
        #M = self.coverage2(X)
        M = X
        #creamos n mapas de cobertutura de cada sensor   
        
        return -np.sum(np.max(M, axis=0))/np.sum(self.sensitivity)
    
    def minimum(self,X):
        """
        
        
        """
    
        M = self.coverage(X)
        #creamos n mapas de cobertutura de cada sensor   
        
        return -np.sum(np.min(M,axis=0))
    
    def xor(self,X):
        """
        
        
        """
    
        M = self.coverage(X)
        
        #XOR
        
        #generamos una mascara indicando los valores que nos interesa tomar en cuenta, los cuales son regiones donde no hay intersección de cobertura
        
        mask = np.sum(M>0,axis=0)==1

        return -(np.sum([mask]*len(M)*M))
    
    def bruteCoverage(self,X):
        """
        
        """
        
        M = self.coverMaps(X)
        
        
        return -np.sum(np.sum(M,axis=0)>0)
    
    def explicability(self,X):
        
        T = np.sum(self.sensitivity)
        E = self.coverage_explicability(X)
        
        C = np.max(E,axis=0)*self.sensitivity
        
        
        
        #creamos n mapas de cobertutura de cada sensor   
        
        return -100*np.sum(C)/T
        
        
        
        
    
    def showPositions(self,X):
        """
        show positions, coverage, histogram, covered sensitivity
        """
        
        n_sensors = int(len(X)/2)
        sensor_list = X.reshape(n_sensors,len(self.NTLI.shape))
        
        positions = np.zeros(np.shape(self.NTLI))
        for i,p in enumerate(sensor_list.astype(int)):
            positions[p[0]][p[1]] = i+1
        return positions
    
    def showVariogram(self,X):
        """
        
        """
        
    def project(self, X):
        """
        
        
        """
        
        dim = self.NTLI.shape
        #returns (mappedSemivariogram from 0 to 1)*EVM
        IMGP = np.copy(self.coverage2(X))
        R = np.zeros((len(IMGP)+1,dim[0],dim[1]))
        dummy = np.ones(dim)*-1
        R[0] = dummy

        for i in range(1, len(R)):
            
            outofrange = (IMGP[i-1]==0)*-1
            inrange = (IMGP[i-1]!=0)*IMGP[i-1]
            R[i] = outofrange+inrange

        return np.argmax(R, axis = 0)*(self.NTLI > 0)
    
    def generateCombinations(self, x, r):
        
        #elements = range(x)

        n = [i for i in range(x)]

        #all the possible combinations without repetition taken in groups  C_(n, x), 
        #n: total number of different elements
        #x: number of selection

        #all the combinations
        return list(it.combinations(n, r))

        #return np.array(combination)
        
        
        
                
        
        
        
        

        
    
 