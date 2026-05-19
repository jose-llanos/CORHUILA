// Reserva.ts
export class reserva {
    id: number = 0;
    vehicleType: string = 'CARRO';
    licensePlate: string = "";
    serviceType: string = "";
    value: number = 0.0;
    reservationDate: string = '';  // Formato: YYYY-MM-DD
    reservationTime: string = '';   // Formato: HH:MM:SS
    active: boolean = true;
}