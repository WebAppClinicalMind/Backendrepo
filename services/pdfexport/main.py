from fpdf import FPDF

def export_to_pdf(df, output_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for i, row in df.iterrows():
        pdf.cell(200, 10, txt=str(row.to_dict()), ln=True)
    pdf.output(output_path)
