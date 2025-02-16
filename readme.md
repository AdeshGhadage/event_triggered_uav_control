# Event-Triggered UAV Control

## Installation

### 1. Clone the Repository

Clone this repository to your local machine using:

```bash
git clone https://github.com/AdeshGhadage/event_triggered_uav_control.git
cd event_triggered_uav_control
```

### 2. Create and Activate a Virtual Environment

It is recommended to use a virtual environment for dependency management:

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file includes libraries such as:
- numpy
- torch
- matplotlib
- scipy


## Running the Simulation

To run the complete simulation, execute the following command:

```bash
python run_simulation.py