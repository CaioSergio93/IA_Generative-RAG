from fpdf import FPDF
import random

def criar_pdf_financeiro(mes, ano):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Título
    pdf.cell(200, 10, txt=f"Relatório Financeiro - {mes}/{ano}", ln=1, align='C')
    pdf.ln(10)
    
    # Dados fictícios
    categorias = ["Vendas", "Serviços", "Assinaturas"]
    receitas = {cat: random.randint(5000, 20000) for cat in categorias}
    
    despesas = {
        "Folha Pagamento": random.randint(15000, 25000),
        "Aluguel": random.randint(5000, 8000),
        "TI": random.randint(3000, 6000)
    }
    
    # Seção de Receitas
    pdf.cell(200, 10, txt="Receitas:", ln=1)
    for cat, val in receitas.items():
        pdf.cell(200, 10, txt=f"- {cat}: R${val:,}", ln=1)
    
    # Seção de Despesas
    pdf.add_page()
    pdf.cell(200, 10, txt="Despesas:", ln=1)
    for cat, val in despesas.items():
        pdf.cell(200, 10, txt=f"- {cat}: R${val:,}", ln=1)
    
    # Saldo
    total_receitas = sum(receitas.values())
    total_despesas = sum(despesas.values())
    saldo = total_receitas - total_despesas
    
    pdf.cell(200, 10, txt=f"\nSaldo Final: R${saldo:,}", ln=1)
    
    # Salvar
    nome_arquivo = f"financeiro_{mes}_{ano}.pdf"
    pdf.output(nome_arquivo)
    return nome_arquivo

# Criar 3 meses de dados
for mes in ["Janeiro", "Fevereiro", "Marco"]:
    criar_pdf_financeiro(mes, 2024)