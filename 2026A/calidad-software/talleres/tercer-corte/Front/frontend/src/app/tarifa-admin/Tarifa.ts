export interface Tarifa {
    id?: number;
    tipoVehiculo: string;
    tarifaDiurna: number;
    tarifaNocturna: number;
    imagen: string; // <-- agrega esta línea
  }