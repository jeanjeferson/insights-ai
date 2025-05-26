from crewai.tools import BaseTool
from typing import Type, Optional
from pydantic import BaseModel, Field, PrivateAttr, field_validator
from langchain_community.tools import DuckDuckGoSearchRun
import time
import json

class DuckSearchInput(BaseModel):
    """Schema otimizado para busca DuckDuckGo com validações robustas."""
    
    query: str = Field(
        ..., 
        description="Termo de busca específico. Use palavras-chave relevantes para joalherias, tendências de mercado, análise competitiva ou contexto econômico.",
        min_length=3,
        max_length=200,
        json_schema_extra={"example": "tendências joalherias 2024 mercado brasileiro"}
    )
    
    domain: Optional[str] = Field(
        "br", 
        description="Domínio para filtrar resultados (br=Brasil, com=Global). Use 'br' para contexto local, 'com' para tendências globais.",
        json_schema_extra={
            "pattern": "^(br|com|org|net)$"
        }
    )
    
    max_results: Optional[int] = Field(
        5,
        description="Número máximo de resultados (1-10). Use 3-5 para análises rápidas, 8-10 para pesquisas aprofundadas.",
        ge=1,
        le=10
    )
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query não pode estar vazia")
        return v.strip()

class DuckDuckGoSearchTool(BaseTool):
    """
    🔍 FERRAMENTA DE PESQUISA WEB INTELIGENTE
    
    QUANDO USAR:
    - Pesquisar tendências de mercado de joalherias
    - Analisar contexto econômico e social
    - Investigar concorrência e benchmarking
    - Buscar insights sobre comportamento do consumidor
    - Validar hipóteses com dados externos
    
    CASOS DE USO ESPECÍFICOS:
    - "Tendências joalherias 2024 Brasil" → Contexto de mercado
    - "Impacto inflação vendas luxo" → Análise econômica
    - "Comportamento consumidor joias online" → Insights de canal
    - "Sazonalidade vendas joalherias natal" → Padrões sazonais
    
    RESULTADOS ENTREGUES:
    - Resumo estruturado de informações relevantes
    - Links para fontes confiáveis
    - Insights contextualizados para o negócio
    - Dados para validação de análises internas
    """
    
    name: str = "DuckDuckGo Search Tool"
    description: str = (
        "Pesquisa web inteligente para contexto de mercado, tendências e análise competitiva. "
        "Use para buscar informações externas que complementem análises internas de vendas. "
        "Ideal para validar hipóteses, entender contexto econômico e identificar tendências de mercado."
    )
    args_schema: Type[BaseModel] = DuckSearchInput
    
    # Atributos privados para controle interno
    _search_tool: DuckDuckGoSearchRun = PrivateAttr(default=None)
    _last_request_time: float = PrivateAttr(default=0.0)
    _MIN_INTERVAL: float = PrivateAttr(default=1.5)  # Rate limiting mais conservador
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._search_tool = DuckDuckGoSearchRun()
        self._last_request_time = 0.0

    def _run(
        self, 
        query: str, 
        domain: str = "br", 
        max_results: int = 5
        ) -> str:
        """
        Executa busca inteligente com rate limiting e formatação estruturada.
        
        Returns:
            JSON estruturado com resultados, insights e metadados
        """
        try:
            # Rate limiting inteligente
            elapsed = time.time() - self._last_request_time
            if elapsed < self._MIN_INTERVAL:
                time.sleep(self._MIN_INTERVAL - elapsed)
            
            self._last_request_time = time.time()
            
            # Construir query otimizada
            if domain and domain != "com":
                search_query = f"{query} site:.{domain}"
            else:
                search_query = query
            
            print(f"🔍 Pesquisando: {search_query}")
            
            # Executar busca
            raw_results = self._search_tool.run(search_query)
            
            # Estruturar resultados
            structured_results = {
                "query_original": query,
                "domain_filter": domain,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results_summary": self._extract_key_insights(raw_results, query),
                "raw_content": raw_results[:1000] + "..." if len(raw_results) > 1000 else raw_results,
                "business_relevance": self._assess_business_relevance(query),
                "recommended_actions": self._generate_recommendations(query, raw_results),
                "metadata": {
                    "search_type": "web_research",
                    "confidence": "medium",
                    "source": "duckduckgo"
                }
            }
            
            return json.dumps(structured_results, ensure_ascii=False, indent=2)
            
        except Exception as e:
            error_response = {
                "error": f"Erro na pesquisa: {str(e)}",
                "query": query,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "fallback_suggestion": f"Tente reformular a busca: '{query}' com termos mais específicos",
                "metadata": {
                    "search_type": "web_research",
                    "status": "error"
                }
            }
            return json.dumps(error_response, ensure_ascii=False, indent=2)
    
    def _extract_key_insights(self, raw_results: str, query: str) -> str:
        """Extrair insights principais dos resultados brutos"""
        # Lógica simples para extrair insights relevantes
        lines = raw_results.split('\n')
        relevant_lines = [line for line in lines if any(keyword in line.lower() 
                         for keyword in ['tendência', 'mercado', 'vendas', 'consumidor', 'crescimento'])]
        
        if relevant_lines:
            return " | ".join(relevant_lines[:3])
        else:
            return raw_results[:200] + "..."
    
    def _assess_business_relevance(self, query: str) -> str:
        """Avaliar relevância para o negócio de joalherias"""
        jewelry_keywords = ['joia', 'joalheria', 'ouro', 'prata', 'anel', 'colar', 'brinco', 'luxo']
        market_keywords = ['mercado', 'vendas', 'consumidor', 'tendência', 'economia']
        
        if any(keyword in query.lower() for keyword in jewelry_keywords):
            return "ALTA - Diretamente relacionado ao setor de joalherias"
        elif any(keyword in query.lower() for keyword in market_keywords):
            return "MÉDIA - Contexto de mercado relevante para análise"
        else:
            return "BAIXA - Informação geral, pode fornecer contexto adicional"
    
    def _generate_recommendations(self, query: str, results: str) -> list:
        """Gerar recomendações baseadas na pesquisa"""
        recommendations = []
        
        if 'tendência' in query.lower():
            recommendations.append("Correlacionar com dados internos de vendas por categoria")
            recommendations.append("Analisar impacto nas projeções de demanda")
        
        if 'mercado' in query.lower():
            recommendations.append("Usar insights para contextualizar análise competitiva")
            recommendations.append("Considerar para ajustes na estratégia de pricing")
        
        if 'consumidor' in query.lower():
            recommendations.append("Integrar com análise de segmentação de clientes")
            recommendations.append("Avaliar oportunidades de novos produtos/serviços")
        
        if not recommendations:
            recommendations.append("Usar informações como contexto adicional nas análises")
        
        return recommendations
