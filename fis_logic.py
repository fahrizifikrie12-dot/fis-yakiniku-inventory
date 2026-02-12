# Fuzzy Inference System for Predicting Purchase Quantity

"""
This implementation is a fuzzy inference system that predicts purchase quantity based on inventory and sales data.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define the fuzzy variables
inventory = ctrl.Variable('inventory', [0, 100, 200, 300, 400, 500])
 sales = ctrl.Variable('sales', [0, 10, 20, 30, 40, 50])
 purchase_quantity = ctrl.Variable('purchase_quantity', [0, 10, 20, 30, 40, 50])

# Define fuzzy membership functions
inventory['low'] = fuzz.trimf(inventory.universe, [0, 0, 250])
inventory['medium'] = fuzz.trimf(inventory.universe, [200, 300, 400])
inventory['high'] = fuzz.trimf(inventory.universe, [350, 500, 500])

sales['low'] = fuzz.trimf(sales.universe, [0, 0, 25])
sales['medium'] = fuzz.trimf(sales.universe, [20, 30, 40])
sales['high'] = fuzz.trimf(sales.universe, [35, 50, 50])

purchase_quantity['low'] = fuzz.trimf(purchase_quantity.universe, [0, 0, 25])
purchase_quantity['medium'] = fuzz.trimf(purchase_quantity.universe, [20, 30, 40])
purchase_quantity['high'] = fuzz.trimf(purchase_quantity.universe, [35, 50, 50])

# Define the rules
rule1 = ctrl.Rule(inventory['low'] & sales['low'], purchase_quantity['high'])
rule2 = ctrl.Rule(inventory['medium'] & sales['medium'], purchase_quantity['medium'])
rule3 = ctrl.Rule(inventory['high'] & sales['high'], purchase_quantity['low'])

# Control system
purchase_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
purchase_simulation = ctrl.ControlSystemSimulation(purchase_ctrl)

# Example usage

# Set the input values
inventory_value = 200
sales_value = 15

purchase_simulation.input['inventory'] = inventory_value
purchase_simulation.input['sales'] = sales_value

# Compute the result
purchase_simulation.compute()

# Get the result
prediction = purchase_simulation.output['purchase_quantity']
print(f'Predicted purchase quantity: {prediction}')