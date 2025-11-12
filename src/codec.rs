use serde_json::Value;
use std::{collections::HashMap, usize, vec};

#[derive(Debug, Clone)]
pub enum DType {
	dt_f32,
	dt_f16,
	dt_bf16,
	dt_f8e5m2,
	dt_f8e4m3,
	dt_i32,
	dt_i16,
	dt_i8,
	dt_u8,
}

fn dtype_to_string(dtype: DType) -> String{
    match dtype {
        DType::dt_bf16 => return "BF16".to_string(),
        DType::dt_f16 => return "F16".to_string(),
        DType::dt_f32 => return "F32".to_string(),
        DType::dt_f8e4m3 => return "F8_E4M3".to_string(),
        DType::dt_f8e5m2 => return "F8_E5M2".to_string(),
        DType::dt_i16 => return "I16".to_string(),
        DType::dt_i32 => return "I32".to_string(),
        DType::dt_i8 => return "I8".to_string(),
        DType::dt_u8 => return "U8".to_string(),
        _ => return "UNKNOWN".to_string()
    }
}

fn dtype_size(dtype: DType) -> usize{
    match dtype {
        DType::dt_bf16 => return 2,
        DType::dt_f16 => return 2,
        DType::dt_f32 => return 4,
        DType::dt_f8e4m3 => return 1,
        DType::dt_f8e5m2 => return 1,
        DType::dt_i16 => return 2,
        DType::dt_i32 => return 4,
        DType::dt_i8 => return 1,
        DType::dt_u8 => return 1,
        _ => return 0
    }
}

#[derive(Debug, Clone)]
pub struct Tensor<'a> {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    data:  &'a[u8],
    size: usize
}


struct YALMData<'a>{
    data: Option<Vec<usize>>,
    size: usize,
    metadata: Option<Value>,
    tensors: HashMap<String, Tensor<'a>>
}

impl<'a> Tensor<'a>{

    pub fn from_json(&mut self, name: String, val: Value, bytes_slice: &'a [u8], bytes_size: usize) -> i32{
        
        self.name = name;       

        let dtype_str: &str = val["dtype"].as_str().unwrap();
        match dtype_str {
            "F32" => self.dtype = DType::dt_f32,
            "F16" => self.dtype = DType::dt_f16,
            "BF16" => self.dtype = DType::dt_bf16,
            "F8_E4M3" => self.dtype = DType::dt_f8e4m3,
            "F8_E5M2" => self.dtype = DType::dt_f8e5m2,
            "I16" => self.dtype = DType::dt_i16,
            "I32" => self.dtype = DType::dt_i32,
            "I8" => self.dtype = DType::dt_i8,
            "U8" => self.dtype = DType::dt_u8,
            _ => eprintln!("Unknown dtype: {}", dtype_str)
        }
        self.size = bytes_size;

        let dsize: usize = dtype_size(self.dtype.clone());
        let mut numel: usize = 1;

        if (val["shape"].as_array().unwrap().len() > 4){
            eprintln!("Shape exceeds 4 dimensions");
        }

        for i in 0..val["shape"].as_array().unwrap().len(){
            if (val["shape"].as_array().unwrap()[i].as_u64().unwrap() as usize != val["shape"].as_array().unwrap()[i]){
                eprintln!("Bad Shape");
                return -1;
            }

            self.shape[i] = val["shape"].as_array().unwrap()[i].as_u64().unwrap() as usize;
            numel *= val["shape"].as_array().unwrap()[i].as_u64().unwrap() as usize ;
        }
        if (val["data_offsets"].as_array().unwrap().len() != 2){
            eprintln!("Bad Data Offsets");
            return -1;
        }

        let offset_start  = val["data_offsets"].as_array().unwrap()[0].as_u64().unwrap() as usize;
        let offset_end  = val["data_offsets"].as_array().unwrap()[1].as_u64().unwrap() as usize;

        if (offset_start < 0 || offset_end < offset_start || offset_end > bytes_size){
            eprintln!("Bad Data Offsets");
        }

        self.data = bytes_slice.get(offset_start..offset_end).ok_or("Slicing failed: bad offsets").unwrap();
        self.size = offset_end - offset_start;

        if (numel * dsize != self.size){
            eprintln!("Bad Size");
            return -1;
        }
        0
 
    }
}

//TODO create YAMLData