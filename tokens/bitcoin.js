import * as bitcoin from "bitcoinjs-lib"; // Import bitcoinjs-lib
import * as bip39 from "bip39"; // Import bip39 for seed generation
import { BIP32Factory } from "bip32";
import * as ecc from "tiny-secp256k1";
import axios from "axios";

let network = bitcoin.networks.testnet;
const { BLOCKSTREAM_TESTNET }  = process.env;
import BitcoinWallets from "../models/bitcoin_wallets.js";


// a function to create bitcoin wallet
export function createBitcoinWallet() {
  // Create the BIP32 instance using the factory and tiny-secp256k1 for elliptic curve operations
  const bip32 = BIP32Factory(ecc);

  // Step 1: Generate a mnemonic phrase (using bip39)
  const mnemonic = bip39.generateMnemonic();

  // Step 2: Generate seed from the mnemonic
  const seed = bip39.mnemonicToSeedSync(mnemonic);

  // Step 3: Use bip32 to derive the HD node from the seed
  const root = bip32.fromSeed(seed, bitcoin.networks.testnet); // Use testnet, or bitcoin.networks.bitcoin for mainnet

  // Step 4: Derive a child key (e.g., first account, first address)
  const account = root.derivePath("m/44'/1'/0'/0/0"); // BIP44 standard path

  // Step 5: Get the private key, public key, WIF, and address
  const privateKey = Array.from( account.privateKey).map((byte) => byte.toString(16)).join(""); // Convert to hex
  const publicKey = Array.from( account.publicKey).map((byte) => byte.toString(16)).join(""); // Convert to hex
  const wif = account.toWIF(); // Wallet Import Format of the private key
  const { address } = bitcoin.payments.p2pkh({
    pubkey: account.publicKey,
    network: bitcoin.networks.testnet, // Or bitcoin.networks.bitcoin for mainnet
  });

  return {
    privateKey,
    publicKey,
    wif,
    address,
    mnemonic,
  };
}




async function getUTXOs(address) {
  try {
    const url = `${BLOCKSTREAM_TESTNET}/api/address/${address}/utxo`;
    const response = await axios.get(url);
    return response.data;
  } catch (error) {
    console.error("Error fetching UTXOs:", error);
  }
}

// a function for sending bitcoin 

export async function sendBitcoin(fromAddress , toAddress, amountSatoshis) {
  const utxos = await getUTXOs(fromAddress); // UTXOs from the previous step

  if (utxos.length === 0) {
    console.log("No UTXOs available to spend.");
    return;
  }

  const txb = new bitcoin.TransactionBuilder(network);

  // 1. Add inputs (UTXOs)
  let totalInput = 0;
  utxos.forEach((utxo) => {
    txb.addInput(utxo.txid, utxo.vout); // Add UTXO as input
    totalInput += utxo.value; // Total value of UTXOs in satoshis
  });

  // 2. Add outputs (recipient and change)
  const fee = 1000; // Set a transaction fee in satoshis
  const change = totalInput - amountSatoshis - fee;

  if (change < 0) {
    console.log("Insufficient balance.");
    return;
  }

  txb.addOutput(toAddress, amountSatoshis); // Recipient address
  txb.addOutput(fromAddress, change); // Change back to the sender

  // 3. Sign the inputs
  utxos.forEach((_, index) => {
    txb.sign(index, keyPair); // Sign each input with the private key
  });

  // 4. Build and broadcast the transaction
  const rawTx = txb.build().toHex();
  console.log("Raw Transaction:", rawTx);

  try {
    const broadcastUrl = `${BLOCKSTREAM_TESTNET}/api/tx`;
    const response = await axios.post(broadcastUrl, rawTx);
    console.log("Transaction ID:", response.data);
  } catch (error) {
    console.error("Error broadcasting transaction:", error);
  }
}

// Call the function to send BTC (testnet)
//sendBitcoin("tb1qljfkdjlq...", 10000); // Send 0.0001 BTC (10000 satoshis)


// a function to get bitcoin balance
export async function getBitcoinBalance(address){ 
  //do your thing
  try {
    const url = `${BLOCKSTREAM_TESTNET}/api/address/${address}`;
    const response = await axios.get(url);
    const balance = response.data.chain_stats.funded_txo_sum - response.data.chain_stats.spent_txo_sum;
    return({
      error: false ,
      balance
    });
  } catch (error) {
    return({
      error: true,
      balance: null,
      message: error.message
    })
    
  }
}


export async function getBitcoinWalletInfo(user) {
  try {
  const { address } = await BitcoinWallets.findOne({ owner: user.id });
  const { balance  , error , message } = await getBitcoinBalance(address);
  if (error ) throw({error , message})
  return {
    balance,
    address,
  };
  }catch(err){ 
      throw (err)
  }
}