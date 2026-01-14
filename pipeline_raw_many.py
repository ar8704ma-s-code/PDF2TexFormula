# src/pipeline/pipeline_raw_many.py

import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any
from detection_ocr import DetectionOCR
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BatchPDFProcessor:
    def __init__(self, device: str = "cuda:0", max_workers: int = None):
        self.device = device
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
        self.processed_count = 0
        self.failed_count = 0
        self.start_time = None
        
    def get_output_directory(self, pdf_path: Path) -> Path:
        """获取输出目录"""
        pdf_name = pdf_path.stem
        out_root = Path("data/Pix2tex_test_1")
        out_root.mkdir(exist_ok=True)

        pdf_output_dir = out_root / pdf_name
        pdf_output_dir.mkdir(exist_ok=True)
        return pdf_output_dir

    def cleanup_intermediate_files(self, output_dir: Path):
        """清理中间文件 - 可选，可以在最后统一清理"""
        # 注释掉或延迟清理以加速处理
        pass

    def validate_results(self, results: Dict[str, Any]) -> bool:
        """验证处理结果"""
        if not results:
            return False
            
        total_formulas = sum(page_data.get("total", 0) for page_data in results.values())
        if total_formulas == 0:
            logger.warning("No formulas detected in PDF")
            return False
            
        return True

    def process_single_pdf(self, pdf_path: Path) -> tuple:
        """处理单个PDF文件 - 修改为返回元组"""
        paper_id = pdf_path.stem
        out_dir = self.get_output_directory(pdf_path)
        final_json = out_dir / "overall_ocr_results.json"
        
        # 检查是否已处理
        if final_json.exists():
            logger.info(f"📁 Skipping {paper_id} - already processed")
            return paper_id, True, "already_processed"
            
        logger.info(f"🔍 Processing {paper_id}")
        
        try:
            # 为每个进程创建新的detector实例，避免CUDA冲突
            detector = DetectionOCR(device=self.device)
            
            # 处理PDF
            results = detector.process_pdf(pdf_path, str(out_dir))
            
            # 验证结果
            if not self.validate_results(results):
                logger.warning(f"⚠️  No valid formulas found in {paper_id}")
                return paper_id, False, "no_formulas"
                
            # 保存最终结果
            final_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
            
            # 延迟清理中间文件
            # self.cleanup_intermediate_files(out_dir)
            
            # 统计信息
            total_pages = len(results)
            total_formulas = sum(page_data.get("total", 0) for page_data in results.values())
            
            logger.info(f"✅ Completed {paper_id} - {total_pages} pages, {total_formulas} formulas")
            return paper_id, True, "success"
            
        except Exception as e:
            logger.error(f"❌ Failed to process {paper_id}: {str(e)}")
            return paper_id, False, str(e)

    def process_batch_parallel(self, pdf_files: List[Path]) -> Dict[str, Any]:
        """并行批量处理PDF文件"""
        self.start_time = time.time()
        total_files = len(pdf_files)
        
        logger.info(f"🚀 Starting parallel processing of {total_files} PDFs with {self.max_workers} workers")
        
        successful = []
        failed = []
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_paper = {
                executor.submit(self.process_single_pdf, pdf_file): pdf_file.stem 
                for pdf_file in pdf_files
            }
            
            # 收集结果
            for i, future in enumerate(as_completed(future_to_paper), 1):
                paper_id = future_to_paper[future]
                try:
                    paper_id, success, message = future.result()
                    if success:
                        successful.append(paper_id)
                        self.processed_count += 1
                    else:
                        failed.append(paper_id)
                        self.failed_count += 1
                except Exception as e:
                    logger.error(f"❌ Unexpected error for {paper_id}: {e}")
                    failed.append(paper_id)
                    self.failed_count += 1
                
                # 进度报告
                if i % 5 == 0 or i == total_files:
                    elapsed = time.time() - self.start_time
                    rate = i / elapsed * 60 if elapsed > 0 else 0
                    logger.info(f"📊 Progress: {i}/{total_files} | "
                               f"Success: {len(successful)} | "
                               f"Failed: {len(failed)} | "
                               f"Rate: {rate:.2f} files/min")
        
        return {
            "successful": successful,
            "failed": failed
        }

    def process_batch_sequential(self, pdf_files: List[Path]) -> Dict[str, Any]:
        """顺序处理PDF文件（原逻辑）"""
        self.start_time = time.time()
        total_files = len(pdf_files)
        
        logger.info(f"🚀 Starting sequential processing of {total_files} PDFs")
        
        successful = []
        failed = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"📄 [{i}/{total_files}] Processing {pdf_file.name}")
            
            paper_id, success, message = self.process_single_pdf(pdf_file)
            if success:
                successful.append(paper_id)
                self.processed_count += 1
            else:
                failed.append(paper_id)
                self.failed_count += 1
            
            # 进度报告
            if i % 10 == 0 or i == total_files:
                elapsed = time.time() - self.start_time
                rate = i / elapsed * 60 if elapsed > 0 else 0
                logger.info(f"📊 Progress: {i}/{total_files} | "
                           f"Success: {len(successful)} | "
                           f"Failed: {len(failed)} | "
                           f"Rate: {rate:.2f} files/min")
        
        return {
            "successful": successful,
            "failed": failed
        }

    def generate_report(self, results: Dict[str, Any], output_dir: Path = Path("output_raw")):
        """生成处理报告"""
        output_dir.mkdir(exist_ok=True)
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": time.time() - self.start_time,
            "total_files_processed": self.processed_count + self.failed_count,
            "successful_processing": self.processed_count,
            "failed_processing": self.failed_count,
            "success_rate": self.processed_count / (self.processed_count + self.failed_count) 
                           if (self.processed_count + self.failed_count) > 0 else 0,
            "successful_papers": results["successful"],
            "failed_papers": results["failed"]
        }
        
        report_file = output_dir / "batch_processing_report.json"
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        
        logger.info(f"📋 Batch processing report saved to {report_file}")
        return report


def main():
    """主函数"""
    # 配置路径
    pdf_root = Path("data/arxiv_math_test/pdf")
    all_pdf_files = sorted(pdf_root.glob("*.pdf"))
    
    if not all_pdf_files:
        logger.error("❌ No PDF files found in data/arxiv_math/pdf/")
        return
    
    logger.info(f"📚 Found {len(all_pdf_files)} PDF files in total")
    

    # 构建需要处理的PDF文件列表
    pdf_files_to_process = all_pdf_files
    logger.info(f"🎯 Filtered to {len(pdf_files_to_process)} new papers to process")
    
    if not pdf_files_to_process:
        logger.info("✅ No new papers to process")
        return
    
    # 显示前几个要处理的论文
    sample_ids = [pdf.stem for pdf in pdf_files_to_process[:5]]
    logger.info(f"📝 Sample papers to process: {', '.join(sample_ids)}")
    
    # 先终止当前作业
    logger.info("🛑 Please terminate the current job first:")
    logger.info("   Run: scancel 1761046")
    
    # 创建处理器
    processor = BatchPDFProcessor(device="cuda:0", max_workers=4)
    
    # 选择处理模式
    use_parallel = True
    
    if use_parallel:
        results = processor.process_batch_parallel(pdf_files_to_process)
    else:
        results = processor.process_batch_sequential(pdf_files_to_process)
    
    # 生成报告
    report = processor.generate_report(results)
    
    # 输出总结
    logger.info("\n" + "="*60)
    logger.info("🎉 BATCH PROCESSING COMPLETED!")
    logger.info(f"✅ Successful: {report['successful_processing']}")
    logger.info(f"❌ Failed: {report['failed_processing']}")
    logger.info(f"📈 Success rate: {report['success_rate']:.1%}")
    logger.info(f"⏱️  Total time: {report['processing_time_seconds']:.1f}s")

if __name__ == "__main__":
    main()