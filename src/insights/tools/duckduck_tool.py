from crewai.tools import BaseTool
from typing import List, Type
from pydantic import BaseModel, Field
from pathlib import Path
from langchain_community.tools import DuckDuckGoSearchRun
import time
import threading

class DuckSearchInput(BaseModel):
    """Esquema de entrada para a ferramenta de busca DuckDuckGo."""
    query: str = Field(..., description="Termo de busca para consultar no DuckDuckGo.")

class DuckDuckGoSearchTool(BaseTool):
    name: str = "DuckDuckGo Search Tool"
    description: str = "Realiza buscas na web utilizando o DuckDuckGo."
    args_schema: Type[BaseModel] = DuckSearchInput
    
    search_tool: DuckDuckGoSearchRun = Field(default_factory=DuckDuckGoSearchRun, exclude=True)
    
    # Controle de rate limit
    _last_request_time: float = 0.0
    _lock = threading.Lock()
    _MIN_INTERVAL = 1.0  # 1 segundo entre requisições

    def _run(self, query: str, domain: str = "br") -> str:
        with self._lock:
            # Calcula tempo desde a última requisição
            elapsed = time.time() - self._last_request_time
            if elapsed < self._MIN_INTERVAL:
                # Espera o tempo restante necessário
                time.sleep(self._MIN_INTERVAL - elapsed)
            
            # Atualiza o tempo da última requisição
            self._last_request_time = time.time()
        
        # Executa a busca
        return self.search_tool.run(f"{query} site:.{domain}")
