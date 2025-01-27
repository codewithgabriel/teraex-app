import { Schema , model } from "mongoose";

const TokenMetaSchema = new Schema({
  label: { type: String, required: true }, 
  logo_url: { type: String, required: true },  
  symbol: { type: String, required: true, unique: true }, 
  
});

const TokensMeta = model("tokens_meta", TokenMetaSchema);
export default TokensMeta;