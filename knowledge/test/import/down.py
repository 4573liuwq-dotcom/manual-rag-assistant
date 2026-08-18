import os

from modelscope import snapshot_download
# BAAI/bge-m3

str = snapshot_download(model_id="BAAI/bge-reranker-large",
                        local_dir="D:/demo")
print(str)
