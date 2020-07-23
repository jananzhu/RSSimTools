from simtk.openmm import app
from simtk import openmm as mm
from simtk.openmm import unit
import mdtraj

aa_residues = "ALA " \
                  "CYS " \
                  "ASP " \
                  "GLU " \
                  "PHE " \
                  "GLY " \
                  "HIS " \
                  "ILE " \
                  "LYS " \
                  "LEU " \
                  "MET " \
                  "ASN " \
                  "PRO " \
                  "GLN " \
                  "ARG " \
                  "SER " \
                  "THR " \
                  "VAL " \
                  "TRP T" \
                  "YR".split()

water_residues = "HOH TIP3 TIP3P SPCE TIP4PEW WAT OH2 TIP".split()

lipid_residues = ['AR', 'CHL', 'DHA', 'LA', 'MY', 'OL', 'PA', 'PC', 'PE', 'PGR', 'PH-', 'PS', 'SA', 'SM', 'ST']


def select_atoms(parmed_structure, selection, ligand_resname=None):
    if ligand_resname:
        lig_residues = ['LIG', ligand_resname]
    else:
        lig_residues = ['LIG']

    protein_chains = [i for i in parmed_structure.topology.chains() if next(i.residues()).name in aa_residues]
    water_chains = [i for i in parmed_structure.topology.chains() if next(i.residues()).name in water_residues]
    ligand_chains = [i for i in parmed_structure.topology.chains() if next(i.residues()).name in lig_residues]
    lipid_chains = [i for i in parmed_structure.topology.chains() if next(i.residues()).name in lipid_residues]

    final_list = []

    if selection[:5] == 'resid':
        resid_list = selection[5:].split()
        chain_selection = resid_list[0]
        resid_list = resid_list[1:]
        #TODO check the order of chains in multichain proteins
        #TODO Currently assuming alphabet designation matches parmed order
        for k,chain in enumerate(protein_chains):
            if ord(chain_selection.lower()-96) == k+1:
                residues = [i for i in chain.residues()]
                selected_residues = [i for i in residues if i.index in resid_list]

                if len(selected_residues) == 0:
                    raise ValueError('Could not find one of the residues {} on protein chain.'.format(resid_list))

                for i in selected_residues:
                    for k in i.atoms():
                        final_list.append(k.index)
    else:
        selection_keywords = selection.split()
        for keyword in selection_keywords:

            if keyword not in ['ligand', 'protein', 'lipids', 'water', 'not']:
                raise ValueError(
                    'Selection could syntax could not be parsed. Options are resid chain int1 int2 int3... ; not; alpha_carbon; protein; ligand; excipients; lipids; water. If seelcting resid cannot use any other keywords')

            if keyword == 'protein':
                # loop across the chains and aggregate all the protein atoms
                for chain in protein_chains:
                    atoms = [i for i in chain.atoms()]
                    atom_indeces = [i.index for i in atoms]
                    for i in atom_indeces:
                        final_list.append(i)

            if keyword == 'water':
                for chain in water_chains:
                    atoms = [i for i in chain.atoms()]
                    atom_indeces = [i.index for i in atoms]
                    for i in atom_indeces:
                        final_list.append(i)

            if keyword == 'ligand':
                for chain in ligand_chains:
                    atoms = [i for i in chain.atoms()]
                    atom_indeces = [i.index for i in atoms]
                    for i in atom_indeces:
                        final_list.append(i)

            if keyword == 'lipid':
                for chain in lipid_chains:
                    atoms = [i for i in chain.atoms()]
                    atom_indeces = [i.index for i in atoms]
                    for i in atom_indeces:
                        final_list.append(i)

            if keyword == 'not':
                all_atom_set = set([i for i in parmed_structure.topology.atoms()])
                inverted_set = all_atom_set - set(final_list)
                final_list = inverted_set

    return final_list


class MDSimulation():
    def __init__(self, parmed_structure, coordinates, 
                 velocities = None, box_vectors = None,
                 temperature = 300 * unit.kelvin, pressure = 1 * unit.atmosphere, membrane = False,
                 integrator = 'Langevin', platform = None, nonbonded_method = 'PME', nonbonded_cutoff = 1 * unit.nanometer,
                 atoms_to_freeze = None, atoms_to_restrain = None,
                 restraint_weight = None, constraints = 'HBonds',
                 log_file = "sim.log",
                 implicit_solvent_model = None, hmr = True,
                 state_report_interval = None, progress_report_interval = None, traj_report_interval = None,
                 traj_fn = 'trajectory.h5'
                 ):

        """
        :param parmed_structure: parmed with topology and force field
        :param parmed: numpy array or hdf5 frame
            #TODO add support for hdf5 frame
        """     
        self.parmed_structure = parmed_structure
        
        #check that the coordinates have units
        if isinstance(coordinates,unit.Quantity):
            self.positions = coordinates
        else:
            raise ValueError('coordinates must have units')
        self.velocities = velocities        
        self.box_vectors = box_vectors
        self.atoms_to_freeze = atoms_to_freeze
        self.platform  = platform
        self.temperature = temperature
        self.integrator = integrator
        self.log_file = log_file
        self.atoms_to_restrain = atoms_to_restrain
        self.restraint_weight = restraint_weight
        self.implicit_solvent_model = implicit_solvent_model
        self.hmr = hmr
        self.constraints = constraints
        self.nonbonded_cutoff = nonbonded_cutoff
        self.nonbonded_method = nonbonded_method
        self.membrane = membrane
        self.pressure = pressure
        self.state_report_interval = state_report_interval
        self.progress_report_interval = progress_report_interval
        self.traj_report_interval = traj_report_interval
        self.log_file = log_file
        self.traj_fn = traj_fn
        self.custom_forces = {}
        
        #if box vectors aren't explicitly specified, attempt to get them from parmed
        if box_vectors == None:
            self.box_vectors = parmed_structure.box_vectors
     
        #run sanity checks
        
        #make sure the parmed has all the params
        if len([i.type for i in parmed_structure.bonds if i.type==None])>0:
            raise ValueError('Parmed object does not have all parameters')
        
        #match num atoms to num coordinates
        if len(parmed_structure.atoms)!=len(self.positions):
            raise ValueError('Number of coordinates does not match number of atoms')
        
        #match num velocities to num coords
        if self.velocities != None and len(self.velocities) != len(self.positions):
            raise ValueError('Number of velocities does not match number of coordinates')
        
        #build sim and add reporters
        self._build_sim()
        self._add_reporters()

    def set_positional_restraints(self, atoms_to_restrain, restraint_weight=None, reference_coordinates=None):
        
        if not reference_coordinates:
            reference_coordinates = self.positions.in_units_of(unit.nanometer)
        else: 
            reference_coordinates = reference_coordinates.in_units_of(unit.nanometer)
        
        if not (isinstance(atoms_to_restrain, list)) and (isinstance(atoms_to_restrain[0],int)):
            raise ValueError('Restraint selection must be a list of integers corresponding to atom indeces')
        if self.restraint_weight==None:
            self.restraint_weight = .2 
            print('Restraint constant not specified, using default .2 kcals/mol*nm')

        restraint_force = mm.CustomExternalForce('K*periodicdistance(x, y, z, x0, y0, z0)^2')
        # Add the restraint weight as a global parameter in kcal/mol/nm^2
        restraint_force.addGlobalParameter("K", self.restraint_weight * unit.kilocalories_per_mole / unit.nanometer ** 2)
        # Define the target xyz coords for the restraint as per-atom (per-particle) parameters
        restraint_force.addPerParticleParameter("x0")
        restraint_force.addPerParticleParameter("y0")
        restraint_force.addPerParticleParameter("z0")

        for index in range(0, len(self.positions)):
            if index in atoms_to_restrain:
                xyz = reference_coordinates[index].in_units_of(unit.nanometers) / unit.nanometers
                restraint_force.addParticle(index, xyz)
        self.custom_forces['positional_restraints']=self.system.addForce(restraint_force)
    
    def update_restraint_weight(self,restraint_weight):
        self.restraint_weight= restraint_weight
        self.simulation.context.setParameter("K",restraint_weight)
    
    def remove_positional_restraints(self):
        self.system.removeForce(self.custom_forces['positional_restraints'])

    def set_implicit_solvent_model(self, implicit_solvent_model, nonbonded_method):
        
        if not "implict_solvent" in self.custom_forces:
            raise ValueError('Implicit solvent model has already been set for this system')
        
        if not implicit_solvent_model in ['HTC', 'OBC1', 'OBC2', 'GBn','GBn2']:
            raise ValueError('Unknown implict solvent model {}'.format(implicit_solvent_model))
            
        implicit_model_object = eval("app.%s" % implicit_solvent_model)
        nonbonded_method_object = eval("app.%s" % nonbonded_method)
        implicit_force = self.parmed_structure.omm_gbsa_force(implicit_model_object,temperature=self.temperature * unit.kelvin,nonbondedMethod=nonbonded_method_object)
        self.custom_forces['implicit_solvent']=self.system.addForce(implicit_force)
        
        
    def freeze_atom_selection(self,atoms_to_freeze):
        
        if not (isinstance(atoms_to_freeze, list)) and (isinstance(atoms_to_freeze[0],int)):
            raise ValueError('Freeze selection must be a list of integers corresponding to atom indeces')
            
        # Set frozen atom masses to zero and then they won't be integrated
        self.frozen_atoms={}
        for atom_index in range(0, len(self.positions)):
            if atom_index in atoms_to_freeze:
                self.frozen_atoms[str(atom_index)]=self.system.getParticleMass(atom_index)
                self.system.setParticleMass(atom_index, 0.0)
        print('froze {} atoms'.format(len(atoms_to_freeze)))

    def unfreeze_atoms(self):
        
        for atom_index in self.frozen_atoms.keys:
            self.system.setParticleMass(int(atom_index), self.frozen_atoms[atom_index])
        print('unfroze {} atoms'.format(len(self.frozen_atoms)))
        del self.frozen_atoms

    def _build_sim(self):
        
        #set step length
        if self.hmr:
            self.step_length = 0.004 * unit.picoseconds
        else:
            self.step_length = 0.002 * unit.picoseconds
            
        if self.constraints in ['HBonds','HAngles','Allbonds']:
            print('setting {} constraints'.format(self.constraints))
            constraints_object = eval("app.%s" % self.constraints)
        else:
            print('did not set any constraints')
            constraints_object=None

        self.system = self.parmed_structure.createSystem(nonbondedMethod=self.nonbonded_method,
                                                 nonbondedCutoff=self.nonbonded_cutoff * unit.angstroms,
                                                 constraints=constraints_object,
                                                 hydrogenMass=4.0 * unit.amu if self.hmr else None)
        

        if self.implicit_solvent_model:
            self.set_implicit_solvent_model(self.implicit_solvent_model,self.nonbonded_cutoff)


        if self.pressure:
            if self.membrane:
                barostat_force = mm.MonteCarloMembraneBarostat(1 * unit.bar, 200 * unit.bar * unit.nanometer, self.temperature * unit.kelvin,
                                                                                  mm.MonteCarloMembraneBarostat.XYIsotropic,
                                                                                  mm.MonteCarloMembraneBarostat.ZFree)
            else:
                barostat_force = mm.MonteCarloBarostat(self.pressure * unit.atmospheres, self.temperature * unit.kelvin, 25)
            self.system.addForce(barostat_force)

            
        if self.atoms_to_restrain is not None:
            self.set_positional_restraints(self.atoms_to_restrain)
            
        if self.atoms_to_freeze is not None:
            self.freeze_atom_selection(self.atoms_to_freeze)


        #get integrator
        if self.integrator not in ['Langevin','Verlet']:
            raise ValueError('{} integrator type not supported'.format(self.integrator))
            
        if self.integrator=='Langevin':    
            self.integrator = mm.LangevinIntegrator(self.temperature * unit.kelvin, 1 / unit.picoseconds, self.step_length)
            
        if self.integrator=='Verlet':
            integrator = mm.VerletIntegrator(self.step_length)

        if self.platform:
            self.platform = mm.Platform.getPlatformByName(self.platform)
            self.simulation = app.Simulation(self.parmed_structure.topology, self.system, self.integrator, platform=self.platform)
        else:
            self.simulation = app.Simulation(self.parmed_structure.topology, self.system, self.integrator)
        
        self.simulation.context.setPositions(self.positions)

        if self.box_vectors is not None:
            self.simulation.context.setPeriodicBoxVectors(self.box_vectors[0], self.box_vectors[1], self.box_vectors[2])

        if self.velocities is not None:
            print('found velocities, restarting from previous simulation')
            self.simulation.context.setVelocities(self.velocities)
        else:
            print('assigning random velocity distribution with temperature {}'.format(self.temperature))
            self.simulation.context.setVelocitiesToTemperature(self.temperature * unit.kelvin)

    def _add_reporters(self):
        if self.state_report_interval:
            state_reporter = app.StateDataReporter(self.log_file, 
                                                   separator="\t",
                                                   reportInterval=self.state_report_interval,
                                                   step=True,
                                                   potentialEnergy=True, 
                                                   kineticEnergy=True, 
                                                   totalEnergy=True,
                                                   volume=True, 
                                                   density=True, 
                                                   temperature=True)

            self.simulation.reporters.append(state_reporter)
    
#         if self.progress_report_interval:
#             progress_reporter = StateDataReporter(stdout,separator="\t",reportInterval=reporter_steps,step=False,totalSteps=totalSteps,time=True,speed=True,progress=True,elapsedTime=False,remainingTime=True)

        if self.traj_report_interval:
            trajectory_steps = int(round(self.traj_report_interval / (self.step_length)))

            traj_reporter = mdtraj.reporters.HDF5Reporter(self.traj_fn, trajectory_steps)
            self.simulation.reporters.append(traj_reporter)

    def run(self, time):
        if not isinstance(time,unit.Quantity):
            raise ValueError ('sim time must have units')
            
        nsteps = int(round(time/self.step_length))
        self.simulation.step(nsteps)


class Minimizer(MDSimulation):

    def run(self, max_steps=None):

        state = self.simulation.context.getState(getPositions=True, getEnergy=True)
        starting_energy = state.getPotentialEnergy()

        print('minimizing system with beginning energy: {}'.format(starting_energy))

        if max_steps is not None:
            self.simulation.minimizeEnergy()
        else:
            self.simulation.minimizeEnergy(maxIterations=max_steps)

        state = self.simulation.context.getState(getPositions=True, getEnergy=True)
        final_energy = state.getPotentialEnergy()

        print('successfully minimized the system to final energy {}'.format(final_energy))
        self.coordinates = state.getPositions()
#
# class MetadynamicsSimulation(MDSimulation):
#     def __init__(self):
        # def super().__init__(CV1=None, CV2=None, CV3 = None)
