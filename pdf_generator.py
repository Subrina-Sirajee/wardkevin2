import os
from fpdf import FPDF

def create_healing_history_pdf(history_records: list, patient_id: str, output_folder: str = "generated_pdfs") -> str:
    """
    Generates a multi-page PDF from a list of historical assessment records.
    This is a stateless function that receives all data it needs.

    Args:
        history_records (list): A list of dictionaries, where each dict is an assessment record.
        patient_id (str): The ID of the patient, used for naming the file.
        output_folder (str): The folder where the generated PDF will be saved.

    Returns:
        str: The file path to the newly created PDF.
    """
    if not history_records:
        raise ValueError("Cannot generate PDF from empty history.")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    pdf_path = os.path.join(output_folder, f"healing_history_{patient_id}_{len(history_records)}_assessments.pdf")
    
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Sort records just in case they are not in order
    sorted_history = sorted(history_records, key=lambda x: x['assessment_date'])

    for i, record in enumerate(sorted_history):
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        
        # --- Page Header ---
        pdf.cell(0, 10, f"Assessment {i + 1} of {len(sorted_history)}", 0, 1, 'C')
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Date: {record['assessment_date']}", 0, 1, 'C')
        pdf.ln(10)

        # --- Image ---
        try:
            # A4 width is 210mm. Center a 100mm wide image.
            pdf.image(record['image_path'], x=(210-100)/2, w=100) 
            pdf.ln(10)
        except Exception as e:
            pdf.set_text_color(255, 0, 0)
            pdf.cell(0, 10, f"Error loading image: {record['image_path']}", 0, 1)
            pdf.set_text_color(0, 0, 0)

        # --- Key Analysis from JSON ---
        analysis = record.get('analysis', {})
        
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Clinical Observations:", 0, 1)
        pdf.set_font("Helvetica", "", 10)
        # Use .encode() to handle potential special characters in the analysis text
        pdf.multi_cell(0, 5, str(analysis.get('Clinical Observations', 'N/A')).encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(5)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Wound Tissue Evaluation:", 0, 1)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, str(analysis.get('Wound Tissue Evaluation', 'N/A')).encode('latin-1', 'replace').decode('latin-1'))
        pdf.ln(5)

    pdf.output(pdf_path)
    print(f"SUCCESS: Generated healing history PDF at: {pdf_path}")
    return pdf_path
