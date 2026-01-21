from batch_pdf_to_md import batch_convert_pdf_to_md
import os

def main():
    print("PDF批量转换工具启动...")
    
    input_dir = "home/robot_psychology/rag/pdf_documents"
    output_dir = "home/robot_psychology/rag/markdown_docs"
    
    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"创建输入目录: {input_dir}")
        os.makedirs(input_dir)
    
    # 显示目录中的PDF文件
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if pdf_files:
        print(f"找到PDF文件: {len(pdf_files)} 个")
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"  {i}. {pdf_file}")
        
        print(f"\n开始批量转换...")
        batch_convert_pdf_to_md(input_dir, output_dir)
        
        # 显示转换结果
        md_files = [f for f in os.listdir(output_dir) if f.endswith('.md')]
        print(f"\n转换完成！生成Markdown文件: {len(md_files)} 个")
        
    else:
        print("请将PDF文件放入 'pdf_documents' 目录")
        print("支持批量处理，可以放入多个PDF文件")

if __name__ == "__main__":
    main()