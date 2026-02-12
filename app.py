from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)
CORS(app)

# Initialize Fuzzy Inference System
def create_fis():
    """Create and return the FIS control system"""
    # Define the fuzzy variables
    inventory = ctrl.Variable('inventory', [0, 100, 200, 300, 400, 500])
    sales = ctrl.Variable('sales', [0, 10, 20, 30, 40, 50])
    purchase_quantity = ctrl.Variable('purchase_quantity', [0, 10, 20, 30, 40, 50])
    
    # Define fuzzy membership functions for inventory
    inventory['low'] = fuzz.trimf(inventory.universe, [0, 0, 250])
    inventory['medium'] = fuzz.trimf(inventory.universe, [200, 300, 400])
    inventory['high'] = fuzz.trimf(inventory.universe, [350, 500, 500])
    
    # Define fuzzy membership functions for sales
    sales['low'] = fuzz.trimf(sales.universe, [0, 0, 25])
    sales['medium'] = fuzz.trimf(sales.universe, [20, 30, 40])
    sales['high'] = fuzz.trimf(sales.universe, [35, 50, 50])
    
    # Define fuzzy membership functions for purchase quantity
    purchase_quantity['low'] = fuzz.trimf(purchase_quantity.universe, [0, 0, 25])
    purchase_quantity['medium'] = fuzz.trimf(purchase_quantity.universe, [20, 30, 40])
    purchase_quantity['high'] = fuzz.trimf(purchase_quantity.universe, [35, 50, 50])
    
    # Define the fuzzy rules
    rule1 = ctrl.Rule(inventory['low'] & sales['low'], purchase_quantity['high'])
    rule2 = ctrl.Rule(inventory['low'] & sales['medium'], purchase_quantity['high'])
    rule3 = ctrl.Rule(inventory['low'] & sales['high'], purchase_quantity['high'])
    rule4 = ctrl.Rule(inventory['medium'] & sales['low'], purchase_quantity['medium'])
    rule5 = ctrl.Rule(inventory['medium'] & sales['medium'], purchase_quantity['medium'])
    rule6 = ctrl.Rule(inventory['medium'] & sales['high'], purchase_quantity['low'])
    rule7 = ctrl.Rule(inventory['high'] & sales['low'], purchase_quantity['low'])
    rule8 = ctrl.Rule(inventory['high'] & sales['medium'], purchase_quantity['low'])
    rule9 = ctrl.Rule(inventory['high'] & sales['high'], purchase_quantity['low'])
    
    # Create control system
    purchase_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    purchase_simulation = ctrl.ControlSystemSimulation(purchase_ctrl)
    
    return purchase_simulation

# Create FIS instance
fis_system = create_fis()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to predict purchase quantity"""
    try:
        data = request.get_json()
        inventory_value = float(data.get('inventory', 0))
        sales_value = float(data.get('sales', 0))
        
        # Validate input ranges
        if not (0 <= inventory_value <= 500):
            return jsonify({'error': 'Inventory harus antara 0-500'}), 400
        if not (0 <= sales_value <= 50):
            return jsonify({'error': 'Penjualan harus antara 0-50'}), 400
        
        # Set input values
        fis_system.input['inventory'] = inventory_value
        fis_system.input['sales'] = sales_value
        
        # Compute the result
        fis_system.compute()
        
        # Get the prediction
        prediction = fis_system.output['purchase_quantity']
        
        return jsonify({
            'success': True,
            'inventory': inventory_value,
            'sales': sales_value,
            'purchase_quantity': round(prediction, 2)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/info', methods=['GET'])
def info():
    """API endpoint to get system info"""
    return jsonify({
        'system': 'Fuzzy Inference System - Yakiniku Like Inventory',
        'version': '1.0.0',
        'description': 'Sistem prediksi jumlah pembelian barang berdasarkan kategori persediaan dan penjualan'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000
)