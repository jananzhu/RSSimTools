class MDSimulation():
    def __init__(self, parmed_structure, coordinates, **kwargs):

        """
        :param parmed_structure: parmed with topology and force field
        :param parmed: numpy array or hdf5 frame
            #TODO add support for hdf5 frame
        """     
        self.parmed_structure = parmed_structure
       
        #set some defaults
        self.opt = {
            'center_box': True,
            'atoms_to_freeze':None,
            'platform':None,
            'temperature': 300 * unit.kelvin,
            'integrator':'Langevin',
            'system_title': 'System',
            'system_id':'0',
            'log_fn':'sim.log',
            'constant_pressure': True,
            'atoms_to_restrain': None,
            'implicit_solvent': False,
            'hmr': True,
            'constraints': True,
            'constraint_type':'HBonds',
            'nonbondedCutoff': 0,
            'velocities': None,
            'box_vectors': None,
            'membrane':False,
            'pressure': 1
            'state_report_interval' : None,
            'progress_report_interval' : None,
            'traj_report_interval' : None
        }
        
        #if any of the defaults are specified overwrite them
        self.opt.update(kwargs)
        print('Initializing simulation using the following params {}'.format(self.opt)) 
        
        self.topology = parmed_structure.topology
        self.box_vectors = parmed_structure.box_vectors
        self.positions = coordinates
        self.velocities = self.opt['velocities']
        
        
        #run sanity checks
        
        #make sure the parmed has all the params
        if len([i.type for i in parmed_structure.bonds if i.type==None])>0:
            raise ValueError('Parmed object does not have all parameters')
        
        #match num atoms to num coordinates
        if len(parmed_structure.atoms)!=len(self.positions):
            raise ValueError('Number of Coordinates does not match Number of Atoms')
        
        #match num velocities to num coords
        if self.opt['velocities'] is not None and len(self.opt['velocities'])!=len(self.positions):
            raise ValueError('Number of Velocities does not match Number of Coordinates')
        

        #build sim
        self._build_sim()
        self._add_reporters()
        
    def _center_box(self):
        center_of_geometry = np.mean(self.positions, axis=0)
        # System box vectors
        box_v = self.box_vectors.in_units_of(unit.nanometer) / unit.angstrom
        box_v = np.array([box_v[0][0], box_v[1][1], box_v[2][2]])
        # Translation vector
        delta = box_v / 2 - center_of_geometry
        # New Coordinates
        self.positions += delta

    def get_implicit_solvent_force(self):
        
        if not self.opt['implicit_model'] in ['HTC', 'OBC1', 'OBC2', 'GBn','GBn2']:
            raise ValueError('Unknown implict model type {}'.format(implicit_model))

        implict_model_object = eval("app.%s" % implicit_model)
        implicit_force = parmed_structure.omm_gbsa_force(implicit_model_object,temperature=self.opt['temperature'] * unit.kelvin,nonbondedMethod=app.PME)
        return implicit_force

    def get_barostat_force(self):
        if self.opt['membrane']:
            barostat_force = mm.MonteCarloMembraneBarostat(1 * unit.bar, 200 * unit.bar * unit.nanometer, self.opt['temperature'] * unit.kelvin,
                                                                              mm.MonteCarloMembraneBarostat.XYIsotropic,
                                                                              mm.MonteCarloMembraneBarostat.ZFree)
        else:
            barostat_force = mm.MonteCarloBarostat(self.opt['pressure'] * unit.atmospheres, self.opt['temperature'] * unit.kelvin, 25)

        return barostat_force

    def get_positonal_restraints_force(self):
        
        if not (isinstance(self.opt['atoms_to_restrain'], list)) and (isinstance(self.opt['atoms_to_restrain'][0],int)):
            raise ValueError('Restraint selection must be a list of integers corresponding to atom indeces')
        if self.opt['restraint_constant']==None:
            force_constant = .2 
            print('restraint constant not specified, using default .2 kcals/mol*nm')
            
        restraint_force = mm.CustomExternalForce('K*periodicdistance(x, y, z, x0, y0, z0)^2')
        # Add the restraint weight as a global parameter in kcal/mol/A^2
        restraint_force.addGlobalParameter("K", self.opt['restraint_constant'] * unit.kilocalories_per_mole / unit.nanometer ** 2)
        # Define the target xyz coords for the restraint as per-atom (per-particle) parameters
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")

        # get the coords in nm
        ref_coords = coords.in_units_of(unit.nanometer)/unit.nanometers
        for index in range(0, len(self.positions)):
            if index in self.opt['atoms_to_restrain']:
                xyz = self.positions[index].in_units_of(unit.nanometers) / unit.nanometers
                restraint_force.addParticle(index, xyz)

        return restraint_force

    def _freeze_atom_selection(self):
        if not (isinstance(self.opt['atoms_to_freeze'], list)) and (isinstance(self.opt['atoms_to_freeze'][0],int)):
            raise ValueError('Freeze selection must be a list of integers corresponding to atom indeces')
        # Set frozen atom masses to zero and then they won't be integrated
        for idx in range(0, len(self.positions)):
            if idx in self.opt['atoms_to_freeze']:
                system.setParticleMass(idx, 0.0)
        return system


    def _build_sim(self):
        
        #set step length
        if self.opt['hmr']:
            self.step_length = 0.004 * unit.picoseconds
        else:
            self.step_length = 0.002 * unit.picoseconds


        if self.opt['constraints']:
            if self.opt['constraint_type'] not in ['HBonds','HAngles','Allbonds']:
                raise ValueError('{} is not a valid constraint type, options are Hbonds, HAngles, or AllBonds'.format(self.opt['constraint_type']))
            constraints = eval("app.%s" % self.opt['constraint_type'])
        else:
            constraints=None

        self.system = parmed_structure.createSystem(nonbondedMethod=app.PME,
                                                 nonbondedCutoff=opt['nonbondedCutoff'] * unit.angstroms,
                                                 constraints=constraints,
                                                 hydrogenMass=4.0 * unit.amu if opt['hmr'] else None)

        if self.opt['center_box']:
            self._center_box()

        if self.opt['implicit_solvent']:
            implicit_solvent_force = self.get_implicit_solvent_force()
            self.system.addForce(implicit_solvent_force)

        if self.opt['constant_pressure']:
            barostat_force = self.get_barostat_force()
            self.system.addForce(barostat_force)
            
        if self.opt['atoms_to_restrain']is not None:
            restraint_force = self.get_positonal_restraints_force()
            self.system.addForce(restraint_force)

        if self.opt['atoms_to_freeze'] is not None:
            self.system = _freeze_atom_selection()


        #get integrator
        if self.opt['integrator'] not in ['Langevin','Verlet']:
            raise ValueError('{} integrator type not supported'.format(self.opt['integrator']))
            
        if self.opt['integrator']=='Langevin':    
            self.integrator = mm.LangevinIntegrator(opt['temperature'] * unit.kelvin, 1 / unit.picoseconds, self.step_length)
            
        if self.opt['integrator']=='Verlet':
            integrator = mm.VerletIntegrator(self.step_length)

        if self.opt['platform']:
            self.platform = mm.Platform.getPlatformByName(self.opt['platform'])
            self.simulation = app.Simulation(self.parmed_structure.topology, self.system, self.integrator, platform=self.platform)
        else:
            self.simulation = app.Simulation(self.parmed_structure.topology, self.system, self.integrator)
        
        self.simulation.context.setPositions(self.positions)

        if self.box_vectors is not None:
            self.simulation.context.setPeriodicBoxVectors(box_vectors[0], box_vectors[1], box_vectors[2])

        if velocities is not None:
            print('Found velocities, restarting from previous simulation')
            self.simulation.context.setVelocities(velocities)
        else:
            print('Starting simulation from random velocity distribution with temperature {}'.format(self.opt['temperature']))
            self.simulation.context.setVelocitiesToTemperature(opt['temperature'] * unit.kelvin)

    def _add_reporters(self)
        if self.opt['state_report_interval']:
            state_reporter = app.StateDataReporter(self.opt['log_fn'], separator="\t",
                                                         reportInterval=reporter_steps,
                                                         step=True,
                                                         potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
                                                         volume=True, density=True, temperature=True)

            self.simulation.reporters.append(state_reporter)
    
        if self.opt['progress_report_interval']:
            progress_reporter = StateDataReporter(stdout,
                                                  separator="\t",
                                                  reportInterval=reporter_steps,
                                                  step=False, totalSteps=totalSteps,
                                                  time=True, speed=True, progress=True,
                                                  elapsedTime=False, remainingTime=True)

        if self.opt['traj_report_interval']:
            trajectory_steps = int(round(opt['trajectory_interval'] / (
                          opt['timestep'].in_units_of(unit.nanoseconds) / unit.nanoseconds)))

            traj_reporter = mdtraj.reporters.HDF5Reporter(opt['trj_fn'], trajectory_steps)
            self.simulation.reporters.append(traj_reporter)

    def run(self, nsteps):
        self.simulation.step(nsteps)