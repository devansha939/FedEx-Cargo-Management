# FedEx 3D Packing Optimization

## Overview
This project addresses the complex 3D packing problem for loading packages into Unit Load Devices (ULDs) for flights. The solution prioritizes efficient utilization of ULDs, adherence to weight and volume constraints, and optimization of cost. The algorithm ensures priority packages are always packed while maintaining high packing efficiency and minimizing costs.

Key outputs of the program include:
1. **Packing Decisions**: Specifies which package is packed in which ULD, along with orientation and coordinates.
2. **Performance Metrics**: Reports costs, packing efficiency, and other relevant statistics.
3. **Visualization**: Generates:
   - A graph illustrating the relationship between cost optimization and volume optimization.
   - A 3D visualization of each ULD with its packed packages.

## Features
- **Greedy Algorithm with Heuristics**: Uses sorting, permutation, and multi-threading to achieve optimal packing and cost minimization.
- **Priority Handling**: Ensures all priority packages are packed, leaving economy packages optional.
- **Cost Optimization**: Balances cost and volume considerations to achieve minimum total cost.
- **Parallel Processing**: Utilizes multi-core processing for improved performance.
- **Visualization**: 
  - Cost vs. Optimization graph.
  - 3D visual representation of packing inside each ULD.
- **Detailed Outputs**:
  - `output.txt`: Packing assignments and details.
  - `costs.txt`: Packing cost metrics.
  - `data.png`: Cost vs. optimization graph.
  - Animated or static 3D views of ULDs with packages.

## Installation
1. Clone this repository.
2. Ensure you have Python 3.8 or higher installed.
3. Install required dependencies:
   ```bash
   pip install matplotlib rectpack
   ```

## Usage
1. Prepare the input file (`input.txt`) containing ULD and package details. The format should be as follows:
   ```plaintext
   <cost of splitting priority packages>
   <number of ULDs>
   <ULD_ID>,<length>,<width>,<height>,<weight_limit>
   ...
   <number of packages>
   <package_ID>,<length>,<width>,<height>,<weight>,<type>,<cost>
   ...
   ```
2. Run the main script:
   ```bash
   python FedEx.py
   ```
3. Generate visualizations:
   - Cost vs. optimization graph is generated automatically and saved as `data.png`.
   - To visualize ULDs with packages in 3D, run:
     ```bash
     python visualize.py
     ```

4. Outputs will be generated:
   - `output.txt`: Packing details (package assignments, ULD coordinates, etc.).
   - `costs.txt`: Costs for each iteration of optimization.
   - `data.png`: Graph showing cost vs. optimization.
   - 3D visualizations of ULDs with packages.

5. GPU Usage
- If you are running on a **GPU**, the threading code may require adjustments depending on the specific library used (e.g., CUDA or OpenCL). You may need to modify the threading parts of the code to ensure the parallel computation can efficiently utilize the GPU's capabilities. This might involve switching from CPU-based multiprocessing to GPU-accelerated frameworks such as TensorFlow, PyTorch, or similar.
- By default, the code assumes a CPU-based environment unless otherwise configured. If using GPU, ensure that the proper libraries (such as cuPy for NumPy-compatible GPU arrays) are installed, and adjust the threading configuration as needed for maximum performance.

## Outputs
### 1. Packing Assignments (`output.txt`)
Lists the ULD each package is assigned to, with orientation and coordinates.

Example:
```plaintext
P-1,U1,0,0,0,20,15,10
P-2,U3,0,0,10,10,10,20
...
```

### 2. Packing Statistics (`costs.txt`)
Includes metrics such as:
- **Cost**: Total minimized cost for each iteration.
- **Cost vs. Optimization Data**: Correlation between optimization parameters and packing cost.

Example:
```plaintext
1: 1600
2: 1500
3: 1550
...
```

### 3. Cost vs. Optimization Graph (`data.png`)
Graph illustrating the relationship between cost and optimization. The minimum cost is highlighted.

### 4. ULD Visualization (3D)
3D visualizations of packages inside ULDs are rendered. Each ULD shows the placement of priority and economy packages, with priority packages visually distinct (black).

## How It Works
1. **Input Parsing**: Reads and parses input data for ULDs and packages.
2. **Package Sorting**:
   - Priority and economy packages sorted by dimensions and cost/volume ratio.
3. **Greedy Packing**:
   - Iterates through ULD permutations.
   - Packs priority packages first, followed by economy packages.
   - Fixes weight constraints and optimizes voids.
4. **Multi-threading**:
   - Divides the workload across available CPU threads for faster execution.
5. **Visualization**:
   - Generates a cost vs. optimization graph using `plotter.py`.
   - Creates 3D representations of ULDs and packages using `visualize.py`.

## Visualization Example
### Graph
![Cost Optimization vs Volume Optimization](data.png)

### ULD Visualization
Each ULD is displayed with packed packages. Priority packages are shown in black, while economy packages are color-coded.
![ULD Packing](packing.png)

## Acknowledgements
This project utilizes:
- **[RectPack](https://github.com/secnot/rectpack)** for 2D rectangle packing.

- **Matplotlib** for visualizing results.
