// 実行前に何らかのPDFファイルをfileステージにアップロードするようにしてください。

select 
    relative_path, 
    build_scoped_file_url(@file, relative_path) as scoped_url, 
    load_pdf(scoped_url) as pdf_text 
from 
    directory(@file);