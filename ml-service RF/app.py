import os
import io
import csv
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from predict import predict_growth

# Firestore setup
try:
    from google.cloud import firestore
    db = firestore.Client()
    FIRESTORE_ENABLED = True
    print("✅ Firestore connected")
except Exception as e:
    db = None
    FIRESTORE_ENABLED = False
    print(f"⚠️ Firestore not available ({e}). Running without database.")

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for the Web Dashboard


@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    required = ['siklus', 'DOC', 'populasi', 'bobot_awal_per_ekor_gr', 'pakan_harian_gr', 'panjang_periode_hari']
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    try:
        result = predict_growth(
            siklus=data['siklus'],
            DOC=data['DOC'],
            populasi=data['populasi'],
            bobot_awal_per_ekor_gr=data['bobot_awal_per_ekor_gr'],
            pakan_harian_gr=data['pakan_harian_gr'],
            panjang_periode_hari=data['panjang_periode_hari']
        )

        # Save to Firestore
        if FIRESTORE_ENABLED:
            doc_data = {
                'input': result['input'],
                'metrics': result['metrics'],
                'predictions': result['predictions'],
                'recommendations': result['recommendations'],
                'timestamp': datetime.utcnow()
            }
            db.collection('fcr').add(doc_data)

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/fcr', methods=['GET'])
def get_fcr():
    """Get prediction history from Firestore."""
    if not FIRESTORE_ENABLED:
        return jsonify({'error': 'Firestore not available'}), 503

    try:
        limit = request.args.get('limit', 50, type=int)
        docs = (db.collection('fcr')
                .order_by('timestamp', direction=firestore.Query.ASCENDING)
                .limit(limit)
                .stream())

        fcr_data = []
        for doc in docs:
            d = doc.to_dict()
            d['id'] = doc.id
            if 'timestamp' in d and d['timestamp']:
                d['timestamp'] = d['timestamp'].isoformat()
            fcr_data.append(d)

        return jsonify({'data': fcr_data, 'count': len(fcr_data)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/export/csv', methods=['GET'])
def export_csv():
    """Export prediction history as a CSV file for Microsoft Excel."""
    if not FIRESTORE_ENABLED:
        return jsonify({'error': 'Firestore not available'}), 503

    try:
        # Get all records or limit
        docs = (db.collection('fcr')
                .order_by('timestamp', direction=firestore.Query.ASCENDING)
                .stream())
                
        # Prepare CSV output in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Headers optimized for Excel reporting
        headers = [
            'ID Riwayat', 'Tanggal Prediksi', 'Siklus', 'DOC', 'Populasi (ekor)', 
            'Bobot Awal/Ekor (gr)', 'Pakan Harian (gr)', 'Periode (hari)',
            'Prediksi Tambah Bobot/Ekor (gr)', 'ADG (gr/ekor/hari)',
            'Estimasi Biomassa Awal (kg)', 'Estimasi Biomassa Akhir (kg)',
            'Prediksi FCR', 'Status Efisiensi', 
            'Rekomendasi Rasio Pakan (%)', 'Aksi Selanjutnya'
        ]
        writer.writerow(headers)
        
        for doc in docs:
            d = doc.to_dict()
            ts = d.get('timestamp')
            date_str = ts.strftime('%Y-%m-%d %H:%M:%S') if ts else ''
            
            i = d.get('input', {})
            m = d.get('metrics', {})
            p = d.get('predictions', {})
            r = d.get('recommendations', {})
            
            row = [
                doc.id,
                date_str,
                i.get('siklus', ''),
                i.get('DOC', ''),
                i.get('populasi', ''),
                i.get('bobot_awal_per_ekor_gr', ''),
                i.get('pakan_harian_gr', ''),
                i.get('panjang_periode_hari', ''),
                
                p.get('delta_bobot_per_ekor_gr', ''),
                m.get('ADG_gr_per_ekor_hari', ''),
                p.get('biomassa_awal_kg', ''),
                p.get('biomassa_akhir_kg', ''),
                
                m.get('FCR', ''),
                m.get('status_efisiensi', ''),
                r.get('rasio_pakan_next_persen', ''),
                r.get('aksi', '')
            ]
            writer.writerow(row)
            
        # Return as downloadable CSV file
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=riwayat_prediksi_pertumbuhan_lele.csv"}
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'lele-growth-prediction-v2',
        'firestore': FIRESTORE_ENABLED
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
