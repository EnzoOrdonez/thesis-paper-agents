# Thesis Paper Agents

Sistema de dos agentes en Python para busqueda automatizada de papers academicos.

Tesis: "Diseno y Validacion de un Modelo Semantico Hibrido para Optimizar Sistemas RAG sobre Documentacion Tecnica Cloud en AWS, Azure y GCP"

## Instalacion

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales
```

## Configuracion

Editar los archivos en `config/`:

- `config.yaml` — Configuracion principal (APIs, directorios, umbrales)
- `keywords.yaml` — Grupos de keywords de busqueda
- `trusted_sources.yaml` — Whitelist de fuentes, categorias, gaps pendientes

Configurar el email en `config.yaml` bajo `apis.openalex.email` y `apis.crossref.email` para acceder al polite pool (mejores rate limits).

## Uso

### Busqueda diaria (Agente 1)

```bash
# Busqueda de los ultimos 7 dias
python daily_researcher.py

# Busqueda de los ultimos 30 dias
python daily_researcher.py --days 30

# Simulacion sin escribir archivos
python daily_researcher.py --dry-run

# Modo daemon (ejecutar diariamente)
python daily_researcher.py --schedule 08:00
```

### Compilador (Agente 2)

```bash
# Pipeline completo de compilacion
python paper_compiler.py

# Revision interactiva de papers nuevos
python paper_compiler.py --review

# Ver estadisticas
python paper_compiler.py --stats

# Exportar referencias
python paper_compiler.py --export-apa
python paper_compiler.py --export-bibtex

# Analisis de gaps
python paper_compiler.py --gap-analysis
```

### Busqueda puntual

```bash
python search_specific.py --title "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
python search_specific.py --author "Khattab" --keyword "ColBERT"
python search_specific.py --doi "10.1145/3397271.3401075"
```

### Agregar paper manualmente

```bash
python add_manual.py --doi "10.1145/3397271.3401075"
python add_manual.py --title "Paper Title" --authors "Author1, Author2" --year 2024 --venue "IEEE"
```

### Pipeline completo

```bash
python run_all.py                    # Busqueda + compilacion
python run_all.py --notify           # Con notificaciones Telegram
python run_all.py --dry-run          # Simulacion
python run_all.py --search-only      # Solo busqueda
python run_all.py --compile-only     # Solo compilacion
python run_all.py --days 14          # Ultimos 14 dias
```

## APIs utilizadas

| API | Proposito | Rate Limit |
|-----|-----------|------------|
| Semantic Scholar | Busqueda principal | 1/seg (10/seg con key) |
| arXiv | Preprints en cs.IR, cs.CL, cs.AI, cs.LG | 1 cada 3 seg |
| OpenAlex | Busqueda + verificacion Scopus | 10/seg con email |
| CrossRef | Validacion de DOIs + metadata | 50/seg con email |

## Estructura de carpetas

```
thesis-paper-agents/
├── config/              # Configuracion YAML
├── src/
│   ├── apis/            # Clientes de API
│   ├── models/          # Modelos Pydantic
│   ├── utils/           # Scoring, dedup, formateo, gaps
│   └── agents/          # Logica de los agentes
├── data/                # Base de datos JSON
├── output/
│   ├── daily/           # Reportes diarios
│   └── reports/         # Reportes consolidados
├── logs/                # Archivos de log
├── daily_researcher.py  # Entry point Agente 1
├── paper_compiler.py    # Entry point Agente 2
├── search_specific.py   # Busqueda puntual
├── add_manual.py        # Agregar paper manual
└── run_all.py           # Pipeline completo
```

## Output

- `output/daily/YYYY-MM-DD_daily_papers.md` — Reporte diario
- `output/reports/consolidated_report.md` — Reporte consolidado por categoria
- `output/reports/gap_analysis.md` — Analisis de gaps
- `output/reports/references_apa7.md` — Referencias APA 7
- `output/reports/references.bib` — BibTeX
- `output/reports/statistics.md` — Estadisticas
