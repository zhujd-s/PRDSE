// Auto-generated BERT encoder workload in the same .m format as the uploaded model files.
// Mapping convention: FC and matrix multiplication operators are represented as CONV operators.
// Softmax, normalization, residual add, GELU and embedding lookup are not modeled because the uploaded Transformer template only models compute-heavy FC/MatMul operators.
// Config: L=6, Seq_Len=128, Hidden=512, Heads=8, Head_Dim=64, FFN=2048.
Constant Seq_Len 128;

Network Bert_small {

	// ----- BERT encoder block 0 -----
	Layer L00_MH_FC_QKV { // QKV projection: batched FC, 512 -> 3*512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 1536, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L00_SD_MatMul_QK_H00 { // Head 0: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L00_SD_MatMul_AV_H00 { // Head 0: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L00_SD_MatMul_QK_H01 { // Head 1: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L00_SD_MatMul_AV_H01 { // Head 1: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L00_SD_MatMul_QK_H02 { // Head 2: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L00_SD_MatMul_AV_H02 { // Head 2: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L00_SD_MatMul_QK_H03 { // Head 3: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L00_SD_MatMul_AV_H03 { // Head 3: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L00_SD_MatMul_QK_H04 { // Head 4: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L00_SD_MatMul_AV_H04 { // Head 4: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L00_SD_MatMul_QK_H05 { // Head 5: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L00_SD_MatMul_AV_H05 { // Head 5: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L00_SD_MatMul_QK_H06 { // Head 6: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L00_SD_MatMul_AV_H06 { // Head 6: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L00_SD_MatMul_QK_H07 { // Head 7: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L00_SD_MatMul_AV_H07 { // Head 7: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L00_MH_FC_AttnOut { // attention output projection: batched FC, 512 -> 512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 512, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L00_FFN_Intermediate { // feed-forward layer A: batched FC, 512 -> 2048
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 2048, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L00_FFN_Output { // feed-forward layer B: batched FC, 2048 -> 512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 512, C: 2048, R: 1, S: 1, Y: 1, X: 1 }
	}


	// ----- BERT encoder block 1 -----
	Layer L01_MH_FC_QKV { // QKV projection: batched FC, 512 -> 3*512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 1536, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L01_SD_MatMul_QK_H00 { // Head 0: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L01_SD_MatMul_AV_H00 { // Head 0: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L01_SD_MatMul_QK_H01 { // Head 1: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L01_SD_MatMul_AV_H01 { // Head 1: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L01_SD_MatMul_QK_H02 { // Head 2: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L01_SD_MatMul_AV_H02 { // Head 2: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L01_SD_MatMul_QK_H03 { // Head 3: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L01_SD_MatMul_AV_H03 { // Head 3: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L01_SD_MatMul_QK_H04 { // Head 4: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L01_SD_MatMul_AV_H04 { // Head 4: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L01_SD_MatMul_QK_H05 { // Head 5: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L01_SD_MatMul_AV_H05 { // Head 5: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L01_SD_MatMul_QK_H06 { // Head 6: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L01_SD_MatMul_AV_H06 { // Head 6: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L01_SD_MatMul_QK_H07 { // Head 7: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L01_SD_MatMul_AV_H07 { // Head 7: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L01_MH_FC_AttnOut { // attention output projection: batched FC, 512 -> 512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 512, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L01_FFN_Intermediate { // feed-forward layer A: batched FC, 512 -> 2048
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 2048, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L01_FFN_Output { // feed-forward layer B: batched FC, 2048 -> 512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 512, C: 2048, R: 1, S: 1, Y: 1, X: 1 }
	}


	// ----- BERT encoder block 2 -----
	Layer L02_MH_FC_QKV { // QKV projection: batched FC, 512 -> 3*512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 1536, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L02_SD_MatMul_QK_H00 { // Head 0: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L02_SD_MatMul_AV_H00 { // Head 0: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L02_SD_MatMul_QK_H01 { // Head 1: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L02_SD_MatMul_AV_H01 { // Head 1: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L02_SD_MatMul_QK_H02 { // Head 2: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L02_SD_MatMul_AV_H02 { // Head 2: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L02_SD_MatMul_QK_H03 { // Head 3: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L02_SD_MatMul_AV_H03 { // Head 3: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L02_SD_MatMul_QK_H04 { // Head 4: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L02_SD_MatMul_AV_H04 { // Head 4: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L02_SD_MatMul_QK_H05 { // Head 5: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L02_SD_MatMul_AV_H05 { // Head 5: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L02_SD_MatMul_QK_H06 { // Head 6: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L02_SD_MatMul_AV_H06 { // Head 6: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L02_SD_MatMul_QK_H07 { // Head 7: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L02_SD_MatMul_AV_H07 { // Head 7: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L02_MH_FC_AttnOut { // attention output projection: batched FC, 512 -> 512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 512, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L02_FFN_Intermediate { // feed-forward layer A: batched FC, 512 -> 2048
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 2048, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L02_FFN_Output { // feed-forward layer B: batched FC, 2048 -> 512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 512, C: 2048, R: 1, S: 1, Y: 1, X: 1 }
	}


	// ----- BERT encoder block 3 -----
	Layer L03_MH_FC_QKV { // QKV projection: batched FC, 512 -> 3*512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 1536, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L03_SD_MatMul_QK_H00 { // Head 0: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L03_SD_MatMul_AV_H00 { // Head 0: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L03_SD_MatMul_QK_H01 { // Head 1: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L03_SD_MatMul_AV_H01 { // Head 1: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L03_SD_MatMul_QK_H02 { // Head 2: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L03_SD_MatMul_AV_H02 { // Head 2: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L03_SD_MatMul_QK_H03 { // Head 3: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L03_SD_MatMul_AV_H03 { // Head 3: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L03_SD_MatMul_QK_H04 { // Head 4: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L03_SD_MatMul_AV_H04 { // Head 4: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L03_SD_MatMul_QK_H05 { // Head 5: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L03_SD_MatMul_AV_H05 { // Head 5: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L03_SD_MatMul_QK_H06 { // Head 6: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L03_SD_MatMul_AV_H06 { // Head 6: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L03_SD_MatMul_QK_H07 { // Head 7: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L03_SD_MatMul_AV_H07 { // Head 7: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L03_MH_FC_AttnOut { // attention output projection: batched FC, 512 -> 512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 512, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L03_FFN_Intermediate { // feed-forward layer A: batched FC, 512 -> 2048
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 2048, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L03_FFN_Output { // feed-forward layer B: batched FC, 2048 -> 512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 512, C: 2048, R: 1, S: 1, Y: 1, X: 1 }
	}


	// ----- BERT encoder block 4 -----
	Layer L04_MH_FC_QKV { // QKV projection: batched FC, 512 -> 3*512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 1536, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L04_SD_MatMul_QK_H00 { // Head 0: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L04_SD_MatMul_AV_H00 { // Head 0: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L04_SD_MatMul_QK_H01 { // Head 1: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L04_SD_MatMul_AV_H01 { // Head 1: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L04_SD_MatMul_QK_H02 { // Head 2: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L04_SD_MatMul_AV_H02 { // Head 2: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L04_SD_MatMul_QK_H03 { // Head 3: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L04_SD_MatMul_AV_H03 { // Head 3: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L04_SD_MatMul_QK_H04 { // Head 4: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L04_SD_MatMul_AV_H04 { // Head 4: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L04_SD_MatMul_QK_H05 { // Head 5: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L04_SD_MatMul_AV_H05 { // Head 5: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L04_SD_MatMul_QK_H06 { // Head 6: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L04_SD_MatMul_AV_H06 { // Head 6: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L04_SD_MatMul_QK_H07 { // Head 7: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L04_SD_MatMul_AV_H07 { // Head 7: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L04_MH_FC_AttnOut { // attention output projection: batched FC, 512 -> 512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 512, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L04_FFN_Intermediate { // feed-forward layer A: batched FC, 512 -> 2048
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 2048, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L04_FFN_Output { // feed-forward layer B: batched FC, 2048 -> 512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 512, C: 2048, R: 1, S: 1, Y: 1, X: 1 }
	}


	// ----- BERT encoder block 5 -----
	Layer L05_MH_FC_QKV { // QKV projection: batched FC, 512 -> 3*512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 1536, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L05_SD_MatMul_QK_H00 { // Head 0: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L05_SD_MatMul_AV_H00 { // Head 0: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L05_SD_MatMul_QK_H01 { // Head 1: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L05_SD_MatMul_AV_H01 { // Head 1: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L05_SD_MatMul_QK_H02 { // Head 2: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L05_SD_MatMul_AV_H02 { // Head 2: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L05_SD_MatMul_QK_H03 { // Head 3: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L05_SD_MatMul_AV_H03 { // Head 3: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L05_SD_MatMul_QK_H04 { // Head 4: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L05_SD_MatMul_AV_H04 { // Head 4: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L05_SD_MatMul_QK_H05 { // Head 5: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L05_SD_MatMul_AV_H05 { // Head 5: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L05_SD_MatMul_QK_H06 { // Head 6: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L05_SD_MatMul_AV_H06 { // Head 6: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L05_SD_MatMul_QK_H07 { // Head 7: Q x K^T, [128x64] x [64x128]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 64, R: 1, S: 1, Y: 1, X: 128 }
	}

	Layer L05_SD_MatMul_AV_H07 { // Head 7: Attention x V, [128x128] x [128x64]
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 1, K: 128, C: 128, R: 1, S: 1, Y: 1, X: 64 }
	}

	Layer L05_MH_FC_AttnOut { // attention output projection: batched FC, 512 -> 512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 512, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L05_FFN_Intermediate { // feed-forward layer A: batched FC, 512 -> 2048
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 2048, C: 512, R: 1, S: 1, Y: 1, X: 1 }
	}

	Layer L05_FFN_Output { // feed-forward layer B: batched FC, 2048 -> 512
		Type: CONV
		Stride { X: 1, Y: 1 }
		Dimensions { N: 128, K: 512, C: 2048, R: 1, S: 1, Y: 1, X: 1 }
	}

}
