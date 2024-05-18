import os
from utils.hausdorff import *
from config import *
import trimesh
from pathlib import Path
from icp_cuda import ICP_Cuda
from icp_cuda.utils import rotate_3d_data

def load_model_by_name(models_dir, model_name, comm):
    models_path = Path(models_dir)
    models = list(map(str, (list(models_path.glob("**/*.off")) + list(models_path.glob("**/*.stl")))))
    filtered = list(filter(lambda x: model_name in x, models))[-1]
    filtered_path = Path(filtered)
    try:
        if not filtered_path.exists():
            raise FileNotFoundError(str(filtered_path))
        model = trimesh.load(str(filtered_path), force="mesh")
    except:
        print_flushed(f"File {model_name} couldn't be loaded. Ensure it exists in the models directory.")
        comm.Abort(1)
    finally:
        if LOAD_OUTPUT:
            print_flushed(f"{model.vertices.shape[0]} vertices is found and loaded by process {comm.Get_rank()}!")
        return model.vertices, model
    
counter = 0
    
def align_model(model, target_model, model_mesh):
    global counter
    icp_cuda = ICP_Cuda(model, target_model)
    icp_cuda.icp(threads_per_block=256, verbose=False)
    transformed_model = icp_cuda.transform()
    model_mesh.vertices = transformed_model
    model_mesh.export(f"transformed_{counter}.off")
    counter += 1
    return transformed_model

def calculate_distance(results_dict, models_dir, fixed_model, model_name, comm, fixed_model_mesh):
    global counter
    print(model_name)
    print(counter)
    model, model_mesh = load_model_by_name(models_dir, model_name, comm)
    model = align_model(model, fixed_model, model_mesh)
    print("MAX:", np.max(model))
    if METHOD == 'SCIPY_DH':
        results_dict[model_name], _, _ = max(directed_hausdorff(fixed_model, model),directed_hausdorff(model, fixed_model))
    elif METHOD == 'EB':
        results_dict[model_name] = max(earlybreak(fixed_model, model),earlybreak(model, fixed_model))
    elif METHOD == 'EB_RS':
        results_dict[model_name] = max(earlybreak_with_rs(fixed_model, model),earlybreak_with_rs(model, fixed_model))
    elif METHOD == 'NAIVEHDD':
        results_dict[model_name] = max(naivehdd(fixed_model, model),naivehdd(model, fixed_model))
    elif METHOD == 'KDTREE':
        results_dict[model_name] = max(kdtree_query(fixed_model, model),kdtree_query(model, fixed_model))
    print(results_dict[model_name])

def print_opening(world_size, models_count, fixed_model_name, alg):
    print_flushed("==================================")
    print_flushed(f"          WORLD SIZE: {world_size}          ")
    print_flushed("==================================")
    if alg == "DLB":
        print_flushed(f"Chosen method: Dynamic Load Balancing — {METHOD}")
    elif alg == "DS":
        print_flushed(f"Chosen method: Static Distribution — {METHOD}")
    print_flushed(f"Total model count: {models_count}")
    print_flushed(f"Fixed model: {fixed_model_name}.")
    print_flushed("-----------------------------------------------")

def print_launch():
    print("* Program should be launched as mpiexec -n <procs> python -m mpi4py main.py and with 3 parameters:\n\
    -- 1st parameter must be either S or D. S for static loading and D for dynamic loading.\n\
    -- 2nd parameter must be the path to the directory with STL or OFF models.\n\
    -- 3rd parameter must be the filename of the model, distance of which to other models will be calculated.'\
** For Example:\n\
    $> mpiexec -n 12 python -m mpi4py main.py D C:models/ModelSet model_1.stl\n\
    $> mpiexec -n 4 python -m mpi4py main.py S D:DataSet/Models airplane_0627.off\n",file=sys.stdout,flush=True)

def print_model_not_exists(directory, model):
    print(f"File {model} does not exist in {directory}",file=sys.stdout,flush=True)