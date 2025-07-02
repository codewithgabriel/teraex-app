import { ethers } from 'ethers'

// Function to create an Ethereum wallet address
export function createEthereumWallet() {
    const wallet = ethers.Wallet.createRandom();
    return {
        address: wallet.address,
        privateKey: wallet.privateKey
    };
}

// Function to send ETH to an address
export async function sendEth(privateKey, toAddress, amount) {
    const provider = ethers.getDefaultProvider('mainnet'); // You can change to 'ropsten', 'rinkeby', etc.
    const wallet = new ethers.Wallet(privateKey, provider);

    const tx = {
        to: toAddress,
        value: ethers.utils.parseEther(amount)
    };

    const transaction = await wallet.sendTransaction(tx);
    return transaction;
}

// Function to get Ethereum balance
export async function getEthereumBalance(address) {
    const provider = ethers.getDefaultProvider('mainnet'); // You can change to 'ropsten', 'rinkeby', etc.
    const balance = await provider.getBalance(address);
    return ethers.utils.formatEther(balance);
}

