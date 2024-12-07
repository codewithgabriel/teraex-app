import { get, post } from "axios";
import * as bitcoin from "bitcoinjs-lib";
let network = bitcoin.networks.testnet;

async function getUTXOs(address) {
  try {
    const url = `https://blockstream.info/testnet/api/address/${address}/utxo`;
    const response = await get(url);
    return response.data;
  } catch (error) {
    console.error("Error fetching UTXOs:", error);
  }
}

async function sendBitcoin(toAddress, amountSatoshis) {
  const utxos = await getUTXOs(address); // UTXOs from the previous step

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
  txb.addOutput(address, change); // Change back to the sender

  // 3. Sign the inputs
  utxos.forEach((_, index) => {
    txb.sign(index, keyPair); // Sign each input with the private key
  });

  // 4. Build and broadcast the transaction
  const rawTx = txb.build().toHex();
  console.log("Raw Transaction:", rawTx);

  try {
    const broadcastUrl = `https://blockstream.info/testnet/api/tx`;
    const response = await post(broadcastUrl, rawTx);
    console.log("Transaction ID:", response.data);
  } catch (error) {
    console.error("Error broadcasting transaction:", error);
  }
}

// Call the function to send BTC (testnet)
sendBitcoin("tb1qljfkdjlq...", 10000); // Send 0.0001 BTC (10000 satoshis)
