from fpdf import FPDF
import random
from datetime import datetime
import numpy as np

class GeradorRelatorios:
    def __init__(self):
        self.categorias_receita = {
            "Vendas": {"min": 8000, "max": 30000, "sazonalidade": [1.1, 1.0, 0.9, 0.8, 0.9, 1.0, 1.1, 1.2, 1.1, 1.0, 1.1, 1.3]},
            "Serviços": {"min": 5000, "max": 20000, "sazonalidade": [1.0]*12},
            "Assinaturas": {"min": 3000, "max": 15000, "sazonalidade": [1.0]*12},
            "Licenciamentos": {"min": 2000, "max": 10000, "sazonalidade": [0.8, 0.8, 0.9, 1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.9, 0.8, 0.8]}
        }
        
        self.categorias_despesa = {
            "Folha Pagamento": {"min": 15000, "max": 25000, "variação": 0.1},
            "Aluguel": {"base": 7000, "variação": 0.05},
            "TI": {"min": 3000, "max": 8000, "variação": 0.15},
            "Marketing": {"min": 2000, "max": 12000, "variação": 0.3},
            "Treinamentos": {"min": 1000, "max": 5000, "variação": 0.2}
        }
        
        self.departamentos = ["Vendas", "Marketing", "TI", "RH", "Financeiro"]
        self.meses_para_numero = {
            "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4,
            "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
            "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12
        }
        
    def _obter_numero_mes(self, mes):
        """Converte nome do mês para número, tratando casos especiais"""
        mes_lower = mes.lower()
        
        # Se for relatório trimestral, pega o mês base
        if mes_lower.startswith("trimestral"):
            mes_base = mes_lower.split()[-1]  # Pega a última palavra
            if mes_base in self.meses_para_numero:
                return self.meses_para_numero[mes_base]
            return random.randint(1, 12)
        
        # Se for mês normal
        if mes_lower in self.meses_para_numero:
            return self.meses_para_numero[mes_lower]
            
        return random.randint(1, 12)
    
    def _gerar_valor_sazonal(self, categoria, mes_num):
        config = self.categorias_receita[categoria]
        base = random.randint(config["min"], config["max"])
        fator = config["sazonalidade"][mes_num-1]
        return int(base * fator)
    
    def _gerar_analise_textual(self, receitas, despesas, saldo):
        analises = []
        
        # Análise de receitas
        maior_cat = max(receitas, key=receitas.get)
        menor_cat = min(receitas, key=receitas.get)
        analises.append(
            f"O destaque do período foi {maior_cat}, responsável por {receitas[maior_cat]/sum(receitas.values()):.1%} da receita total. "
            f"Enquanto isso, {menor_cat} apresentou o menor desempenho."
        )
        
        # Análise de despesas
        maior_desp = max(despesas, key=despesas.get)
        analises.append(
            f"Em relação às despesas, {maior_desp} consumiu {despesas[maior_desp]/sum(despesas.values()):.1%} do total. "
            f"Este valor está {'acima' if random.random() > 0.5 else 'dentro'} da média histórica."
        )
        
        # Análise de saldo
        if saldo > 0:
            analises.append(
                f"O período encerrou com superávit de R${saldo:,}, indicando uma saúde financeira positiva. "
                f"Recomenda-se reinvestir {random.randint(20, 50)}% deste valor em {random.choice(['infraestrutura', 'inovações', 'capacitação'])}."
            )
        else:
            analises.append(
                f"O período encerrou com déficit de R${abs(saldo):,}. "
                f"Sugere-se revisão nos gastos com {random.choice(list(despesas.keys()))} e reavaliação estratégica."
            )
        
        return "\n".join(analises)
    
    def _gerar_metadados(self, mes, ano):
        return {
            "autor": "Departamento Financeiro",
            "data_geracao": datetime.now().strftime("%Y-%m-%d"),
            "versao": "1.0",
            "tipo": "Relatório Trimestral" if "trimestral" in mes.lower() else "Relatório Mensal",
            "mes": mes,
            "ano": ano,
            "confidencial": random.choice(["Interno", "Confidencial", "Restrito"])
        }
    
    def gerar_relatorio(self, mes, ano):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        # Cabeçalho
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt=f"Relatório Financeiro - {mes}/{ano}", ln=1, align='C')
        pdf.ln(10)
        
        # Metadados (não visíveis no PDF)
        metadados = self._gerar_metadados(mes, ano)
        
        # Receitas
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Receitas", ln=1)
        pdf.set_font("Arial", size=12)
        
        num_mes = self._obter_numero_mes(mes)
        receitas = {cat: self._gerar_valor_sazonal(cat, num_mes) for cat in self.categorias_receita}
        
        for cat, val in receitas.items():
            pdf.cell(200, 8, txt=f"- {cat}: R${val:,}", ln=1)
        pdf.cell(200, 8, txt=f"Total Receitas: R${sum(receitas.values()):,}", ln=1)
        pdf.ln(5)
        
        # Despesas
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Despesas", ln=1)
        pdf.set_font("Arial", size=12)
        
        despesas = {}
        for cat, config in self.categorias_despesa.items():
            if "base" in config:
                base = config["base"]
                variacao = random.uniform(-config["variação"], config["variação"])
                despesas[cat] = int(base * (1 + variacao))
            else:
                despesas[cat] = random.randint(config["min"], config["max"])
        
        for cat, val in despesas.items():
            pdf.cell(200, 8, txt=f"- {cat}: R${val:,}", ln=1)
        pdf.cell(200, 8, txt=f"Total Despesas: R${sum(despesas.values()):,}", ln=1)
        pdf.ln(5)
        
        # Saldo
        saldo = sum(receitas.values()) - sum(despesas.values())
        pdf.cell(200, 8, txt=f"Saldo Final: R${saldo:,}", ln=1)
        pdf.ln(10)
        
        # Análise
        pdf.set_font("Arial", 'I', 12)
        pdf.multi_cell(0, 8, txt=self._gerar_analise_textual(receitas, despesas, saldo))
        
        # Detalhes por departamento
        pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt="Despesas por Departamento", ln=1)
        pdf.set_font("Arial", size=12)
        
        desp_departamentos = {depto: random.randint(2000, 15000) for depto in self.departamentos}
        for depto, val in desp_departamentos.items():
            pdf.cell(200, 8, txt=f"- {depto}: R${val:,}", ln=1)
        
        # Salvar
        nome_arquivo = f"relatorio_financeiro_{mes.lower().replace(' ', '_')}_{ano}.pdf"
        pdf.output(nome_arquivo)
        
        # Adicionar metadados ao nome do arquivo (para teste RAG)
        with open(nome_arquivo, 'ab') as f:
            f.write(f"\n\n<!-- METADADOS: {str(metadados)} -->".encode('utf-8'))
        
        return nome_arquivo

# Gerar relatórios de teste
gerador = GeradorRelatorios()

# Últimos 12 meses
meses = [
    "Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho",
    "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"
]
ano = 2024

for mes in meses:
    # Gerar com variação anual (2023 e 2024)
    gerador.gerar_relatorio(mes, ano - random.randint(0, 1))
    
    # Gerar alguns trimestrais a cada 3 meses
    if random.random() > 0.7:  # 30% de chance de gerar trimestral
        gerador.gerar_relatorio(f"Trimestral {mes}", ano - random.randint(0, 1))

print("✅ Relatórios gerados com sucesso!")